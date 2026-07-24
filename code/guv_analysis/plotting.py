import os
import csv
from glob import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bioio import BioImage
from scipy.signal import find_peaks, peak_widths
import tifffile as tif
from tqdm import tqdm
from image import crop_around_guv

def plot_profile_and_zoom(
    channels_data,
    ves_coordinates,
    image_dim,
    radial_profiles,
    along_radius,
    angular_profiles=None,
    size_view=1.5,
    channels=(0, 1, 2),
    save_path=None,
    show=True
):
    """
    Plot cropped channel images stacked vertically, plus radial and angular profiles.

    No overlay.
    No circle.
    No distance normalization.
    """

    num_channels_to_show = len(channels)

    if angular_profiles is None:
        num_columns = 2
    else:
        num_columns = 3

    fig, axes = plt.subplots(
        num_channels_to_show,
        num_columns,
        figsize=(4 * num_columns, 3 * num_channels_to_show)
    )

    if num_channels_to_show == 1:
        axes = np.array([axes])

    for row, ch in enumerate(channels):
        crop = crop_around_guv(
            channels_data[ch, :, :],
            size_view,
            image_dim,
            ves_coordinates
        )

        axes[row, 0].imshow(crop, cmap="gray")
        axes[row, 0].set_title(f"Channel {ch} crop")
        axes[row, 0].axis("off")

        axes[row, 1].plot(along_radius, radial_profiles[:, ch])
        axes[row, 1].set_title(f"Channel {ch} radial profile")
        axes[row, 1].set_xlabel("Distance from center (px)")
        axes[row, 1].set_ylabel("Intensity")

        if angular_profiles is not None:
            axes[row, 2].plot(angular_profiles[:, ch])
            axes[row, 2].set_title(f"Channel {ch} angular profile")
            axes[row, 2].set_xlabel("Angle index")
            axes[row, 2].set_ylabel("Intensity")

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

def save_separate_profile_plots(
    radial_profiles,
    along_radius,
    angular_profiles,
    vesicle_id,
    channels,
    output_folder
):
    """
    Save separate radial and angular profile plots for each channel.

    Files are named:
        radial_ves{vesicle_id}_ch{ch}.png
        angular_ves{vesicle_id}_ch{ch}.png
    """

    for ch in channels:

        # Radial profile
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(along_radius, radial_profiles[:, ch])
        ax.set_title(f"Vesicle {vesicle_id} - channel {ch} radial profile")
        ax.set_xlabel("Distance from center (px)")
        ax.set_ylabel("Intensity")
        plt.tight_layout()

        radial_path = os.path.join(
            output_folder,
            f"radial_ves{vesicle_id}_ch{ch}.png"
        )

        fig.savefig(radial_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        # Angular profile
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(angular_profiles[:, ch])
        ax.set_title(f"Vesicle {vesicle_id} - channel {ch} angular profile")
        ax.set_xlabel("Angle index")
        ax.set_ylabel("Intensity")
        plt.tight_layout()

        angular_path = os.path.join(
            output_folder,
            f"angular_ves{vesicle_id}_ch{ch}.png"
        )

        fig.savefig(angular_path, dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_detected_guv_shape(
    channel_data,
    ves_coordinates,
    image_dim,
    shape_x,
    shape_y,
    size_view=1.5,
    title="Detected GUV shape",
    save_path=None,
    show=True
):
    """
    Plot the detected GUV shape on top of a cropped channel image.
    """

    xc = ves_coordinates[1]
    yc = ves_coordinates[2]
    radius = ves_coordinates[3]

    crop = crop_around_guv(
        channel_data,
        size_view,
        image_dim,
        ves_coordinates
    )

    vesicle_box_side = int(size_view * radius)

    move_right = 0
    move_left = 0
    move_up = 0
    move_down = 0

    if xc - vesicle_box_side < 0:
        move_right = vesicle_box_side - xc

    if xc + vesicle_box_side > image_dim[0]:
        move_left = vesicle_box_side - (image_dim[0] - xc)

    if yc - vesicle_box_side < 0:
        move_down = vesicle_box_side - yc

    if yc + vesicle_box_side > image_dim[1]:
        move_up = vesicle_box_side - (image_dim[1] - yc)

    x_min = int(xc - vesicle_box_side + move_right - move_left)
    y_min = int(yc - vesicle_box_side + move_down - move_up)

    shape_x_crop = shape_x - x_min
    shape_y_crop = shape_y - y_min

    fig, ax = plt.subplots(figsize=(5, 5))

    ax.imshow(crop, cmap="gray")
    ax.plot(shape_x_crop, shape_y_crop, ".", markersize=4, color="magenta")
    ax.set_title(title)
    ax.axis("off")

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_shape_normalized_profiles_crops_and_angles(
    channels_data,
    ves_coordinates,
    image_dim,
    radial_profiles_normalized,
    normalized_distance_axis,
    angular_profiles_normalized,
    channels=(0, 1, 2),
    size_view=1.5,
    title="Shape-normalized profiles",
    save_path=None,
    show=True
):
    """
    Plot cropped channel images, shape-normalized radial profiles,
    and shape-normalized angular profiles.

    Rows correspond to channels.

    Columns are:
        1. cropped image around the GUV
        2. radial profile normalized by local membrane position
        3. angular profile around the detected membrane
    """

    num_channels_to_show = len(channels)

    fig, axes = plt.subplots(
        num_channels_to_show,
        3,
        figsize=(13, 3 * num_channels_to_show)
    )

    if num_channels_to_show == 1:
        axes = np.array([axes])

    fig.suptitle(title)

    angle_index = np.arange(
        angular_profiles_normalized.shape[0]
    )

    for row, ch in enumerate(channels):

        crop = crop_around_guv(
            channels_data[ch, :, :],
            size_view,
            image_dim,
            ves_coordinates
        )

        # Column 1: crop
        axes[row, 0].imshow(crop, cmap="gray")
        axes[row, 0].set_title(f"Channel {ch} crop")
        axes[row, 0].axis("off")

        # Column 2: radial profile
        axes[row, 1].plot(
            normalized_distance_axis,
            radial_profiles_normalized[:, ch]
        )

        axes[row, 1].axvline(
            1,
            linestyle="--",
            linewidth=1,
            label="Detected membrane"
        )

        axes[row, 1].set_title(f"Channel {ch} radial profile")
        axes[row, 1].set_xlabel("Normalized distance")
        axes[row, 1].set_ylabel("Intensity")
        axes[row, 1].legend()

        # Column 3: angular profile
        axes[row, 2].plot(
            angle_index,
            angular_profiles_normalized[:, ch]
        )

        axes[row, 2].set_title(f"Channel {ch} angular profile")
        axes[row, 2].set_xlabel("Angle index")
        axes[row, 2].set_ylabel("Membrane intensity")

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)