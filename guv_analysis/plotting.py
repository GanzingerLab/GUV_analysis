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
from .image_tools import crop_around_guv

def plot_profile_and_zoom(
    channels_data,
    ves_coordinates,
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
    num_columns = 2 if angular_profiles is None else 3

    fig, axes = plt.subplots(
        num_channels_to_show,
        num_columns,
        figsize=(4 * num_columns, 3 * num_channels_to_show),
    )

    if num_channels_to_show == 1:
        axes = np.array([axes])

    # Crop every channel using the same boundaries
    cropped_channels = crop_around_guv(
        channels_data,
        size_view,
        ves_coordinates,
    )

    for row, ch in enumerate(channels):
        axes[row, 0].imshow(cropped_channels[ch], cmap="gray")
        axes[row, 0].set_title(f"Channel {ch} crop")
        axes[row, 0].axis("off")

        axes[row, 1].plot(
            along_radius,
            radial_profiles[:, ch],
        )
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


def plot_detected_guv_shape(img, channel, ves_coordinates, shape_x, shape_y, size_view=1.5, title=None):
    """Show a detected GUV membrane shape over the selected image channel."""
    _, xc, yc, radius = ves_coordinates
    view_radius = size_view * radius
    x_min = max(0, int(xc - view_radius))
    x_max = min(img.shape[2], int(xc + view_radius + 1))
    y_min = max(0, int(yc - view_radius))
    y_max = min(img.shape[1], int(yc + view_radius + 1))

    plt.figure()
    plt.imshow(img[channel, y_min:y_max, x_min:x_max], cmap="gray")
    plt.plot(shape_x - x_min, shape_y - y_min)
    plt.scatter(xc - x_min, yc - y_min, marker="+")
    plt.title(title or "Detected GUV shape")
    plt.axis("equal")
    plt.show()


def plot_shape_normalized_profiles_crops_and_angles(
    channels_data,
    ves_coordinates,
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

    cropped_channels = crop_around_guv(
    channels_data,
    size_view,
    ves_coordinates,
    )

    for row, ch in enumerate(channels):

        crop = cropped_channels[ch]

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