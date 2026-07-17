# %% Imports
import os
import csv
from glob import glob
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bioio import BioImage
from scipy.ndimage import uniform_filter1d
import tifffile as tif
from tqdm import tqdm

# %% Global parameters and functions
GUV_CH = 0
DETECTION_SUFFIX = "_detected_vesicles.csv"

PATH = r"D:\Data\EVOLF"
IMAGE_FORMAT = '.tif'
num_angles = 120
length_excess = 1.5
dr = 1

SIZE_VIEW = length_excess

INNER_MARGIN = 1.2
OUTER_MARGIN = 2.0

PARAMETERS_PROFILES = np.array((num_angles, length_excess, dr))

SMOOTH_WINDOW = 5

DP_SMOOTHNESS_WEIGHT = 0.5
DP_RADIUS_PRIOR_WEIGHT = 0.001
DP_MIN_RADIUS_FRACTION = 0.5
DP_MAX_RADIUS_FRACTION = 1.5
DP_FINAL_SMOOTHING_WINDOW = 9

MEMBRANE_NORM_IN = 0.95
MEMBRANE_NORM_OUT = 1.05

MIN_VALID_SHAPE_FRACTION = 0.9

PLOT_RESULTS = True

pixels_to_remove = 2
size_central_area = 1 / 4

#%%
# Image loading and saving
def open_czi(path):
    img_bio = BioImage(path)
    img = img_bio.get_image_data()

    img, dims = squeeze_singleton_dims(img, img_bio.dims.order)

    return img

def open_tif(path):
    img = tif.imread(path)
    return img

def open_image(path):
    if path[-4:] == '.tif':
        return open_tif(path)
    if path[-4:] == '.czi':
        return open_czi(path)

def get_image_stem(image_path):
    image_name = os.path.basename(image_path)
    image_stem = os.path.splitext(image_name)[0]

    return image_stem

def squeeze_singleton_dims(img, dims):
    keep_dims = []

    for size, dim in zip(img.shape, dims):
        if size != 1:
            keep_dims.append(dim)

    img_squeezed = np.squeeze(
        img,
        axis=tuple(axis for axis, size in enumerate(img.shape) if size == 1)
    )

    dims_squeezed = "".join(keep_dims)

    return img_squeezed, dims_squeezed


def get_output_folder(image_path, base_path):
    relative_path = os.path.relpath(image_path, base_path)
    relative_folder = os.path.dirname(relative_path)

    output_folder = os.path.join(base_path, "output", relative_folder)

    os.makedirs(output_folder, exist_ok=True)

    return output_folder


def linear_profiles(channels_data, ves_coordinates, image_dim, parameters_profiles):
    """
    Calculate linear radial intensity profiles for one vesicle.
    """

    num_angles = int(parameters_profiles[0])
    length_excess = parameters_profiles[1]
    dr = parameters_profiles[2]

    xc = ves_coordinates[1]
    yc = ves_coordinates[2]
    radius = ves_coordinates[3]

    num_channels = channels_data.shape[0]

    profile_radius_limit = length_excess * radius

    theta = np.linspace(0, 2 * np.pi, num_angles, endpoint=False)
    along_radius = np.arange(0, int(profile_radius_limit), dr)

    intensity_profiles = np.zeros(
        (
            len(along_radius),
            num_angles,
            num_channels
        )
    )

    death_mark = False

    for angle_i, angle in enumerate(theta):

        line_x = xc + along_radius * np.cos(angle)
        line_y = yc + along_radius * np.sin(angle)

        if (
            np.any(line_x < 0)
            or np.any(line_x >= image_dim[0])
            or np.any(line_y < 0)
            or np.any(line_y >= image_dim[1])
        ):
            death_mark = True
            break

        line_x = np.round(line_x).astype(int)
        line_y = np.round(line_y).astype(int)

        for ch in range(num_channels):
            intensity_profiles[:, angle_i, ch] = channels_data[
                ch,
                line_y,
                line_x
            ]

    return intensity_profiles, along_radius, theta, death_mark


def background_noise(
    channels_data,
    ves_coordinates,
    inner_margin=1.2,
    outer_margin=2.0
):
    """
    Estimate local background around one vesicle.

    Background is calculated from an annulus around the vesicle.
    """

    xc = ves_coordinates[1]
    yc = ves_coordinates[2]
    radius = ves_coordinates[3]

    num_channels = channels_data.shape[0]
    image_y = channels_data.shape[1]
    image_x = channels_data.shape[2]

    yy, xx = np.meshgrid(
        np.arange(image_y),
        np.arange(image_x),
        indexing="ij"
    )

    distance_from_center = np.sqrt(
        (xx - xc) ** 2
        + (yy - yc) ** 2
    )

    background_mask = (
        (distance_from_center >= inner_margin * radius)
        & (distance_from_center <= outer_margin * radius)
    )

    background = np.zeros((1, num_channels))

    for ch in range(num_channels):
        background[0, ch] = np.nanmean(channels_data[ch, :, :][background_mask])

    return background


def background_correction(num_channels, intensity_profiles, background):
    """
    Subtract background from intensity profiles.
    """

    intensity_profiles_corrected = np.zeros_like(intensity_profiles)

    for i in range(num_channels):
        intensity_profiles_corrected[:, :, i] = np.clip(
            intensity_profiles[:, :, i] - background[0, i],
            0,
            None
        )

    return intensity_profiles_corrected


def radial_profile(num_channels, intensity_profiles):
    """
    Calculate average radial profile over all angular profiles.
    """

    radial_profiles = np.zeros(
        (
            intensity_profiles.shape[0],
            num_channels
        )
    )

    for i in range(num_channels):
        radial_profiles[:, i] = np.average(
            intensity_profiles[:, :, i],
            axis=1
        )

    return radial_profiles


def zoom_in_vesicle(channel, size_side, image_dim, ves_coordinates):
    """
    Crop a square image around one vesicle.
    """

    xc = ves_coordinates[1]
    yc = ves_coordinates[2]
    radius = ves_coordinates[3]

    vesicle_box_side = int(size_side * radius)

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
    x_max = int(xc + vesicle_box_side + move_right - move_left)

    y_min = int(yc - vesicle_box_side + move_down - move_up)
    y_max = int(yc + vesicle_box_side + move_down - move_up)

    crop = channel[y_min:y_max, x_min:x_max]

    return crop


def circular_rolling_average_profiles(intensity_profiles, window_size=5):
    """
    Smooth intensity profiles by circular rolling average along the angular axis.

    The rolling average is applied across neighboring linear profiles/angles.
    The angular axis is treated as circular, so the first and last profiles are
    connected.

    Input
    -----
    intensity_profiles : np.ndarray
        Intensity profiles with shape:
            radial position x angle/profile index x channel

    window_size : int
        Number of neighboring angular profiles to average.
        Must be an odd number. For example, window_size=5 averages:
            2 profiles before, current profile, 2 profiles after.

    Returns
    -------
    intensity_profiles_smoothed : np.ndarray
        Smoothed intensity profiles with the same shape as intensity_profiles.
    """

    if window_size % 2 == 0:
        raise ValueError("window_size must be odd.")

    half_window = window_size // 2

    intensity_profiles_smoothed = np.zeros_like(intensity_profiles)

    for shift in range(-half_window, half_window + 1):
        intensity_profiles_smoothed = (
            intensity_profiles_smoothed
            + np.roll(intensity_profiles, shift=shift, axis=1)
        )

    intensity_profiles_smoothed = intensity_profiles_smoothed / window_size

    return intensity_profiles_smoothed


def circular_smooth_peak_positions(peak_positions, window_size=7):
    """
    Smooth detected membrane positions along the angular direction.

    The angular coordinate is circular, so the first and last angles are treated
    as neighbors.
    """

    valid = ~np.isnan(peak_positions)

    if np.sum(valid) < 3:
        return peak_positions

    peak_positions_filled = peak_positions.copy()

    num_angles = len(peak_positions)
    angle_index = np.arange(num_angles)

    valid_index = angle_index[valid]
    valid_values = peak_positions[valid]

    extended_index = np.concatenate([
        valid_index - num_angles,
        valid_index,
        valid_index + num_angles
    ])

    extended_values = np.concatenate([
        valid_values,
        valid_values,
        valid_values
    ])

    peak_positions_filled = np.interp(
        angle_index,
        extended_index,
        extended_values
    )

    peak_positions_smooth = uniform_filter1d(
        peak_positions_filled,
        size=window_size,
        mode="wrap"
    )

    return peak_positions_smooth


def detect_membrane_shape_dp(
    intensity_profiles_smooth,
    along_radius,
    theta,
    ves_coordinates,
    channel=0,
    smoothness_weight=0.15,
    radius_prior_weight=0.01,
    min_radius_fraction=0.35,
    max_radius_fraction=1.8,
    final_smoothing_window=7
):
    """
    Detect the GUV membrane as a globally smooth radial path.

    The selected membrane
    position is chosen by balancing:
        1. high membrane-channel intensity
        2. smoothness between neighboring angles
        3. weak preference for the approximate detected radius

    This allows deformed or elongated GUVs because the radius prior is weak.

    Input
    -----
    intensity_profiles_smooth : np.ndarray
        Smoothed intensity profiles with shape:
            radial position x angle/profile index x channel

    along_radius : np.ndarray
        Distance values along each radial profile.

    theta : np.ndarray
        Angle values corresponding to the angular profiles.

    ves_coordinates : np.ndarray
        Vesicle coordinates:
            ves_coordinates[0] : vesicle id
            ves_coordinates[1] : xc
            ves_coordinates[2] : yc
            ves_coordinates[3] : approximate radius

    channel : int
        Channel used for membrane detection.

    smoothness_weight : float
        Penalty for sudden changes in radius between neighboring angles.
        Increase this to force a smoother contour.

    radius_prior_weight : float
        Weak penalty for being far from the approximate detected radius.
        Keep this small for deformed GUVs.

    min_radius_fraction : float
        Minimum allowed radius relative to the approximate radius.

    max_radius_fraction : float
        Maximum allowed radius relative to the approximate radius.

    final_smoothing_window : int
        Circular smoothing window applied to the final detected radius sequence.

    Returns
    -------
    shape_x : np.ndarray
        X coordinates of the detected membrane shape.

    shape_y : np.ndarray
        Y coordinates of the detected membrane shape.

    peak_positions : np.ndarray
        Selected radial membrane position for each angle.

    peak_found : np.ndarray
        Boolean array. Here it is True for every angle in the optimized path.
    """

    #extract center and radius from DisGUVery
    xc = ves_coordinates[1]
    yc = ves_coordinates[2]
    approx_radius = ves_coordinates[3]

    #extract membrane channel
    profile_image = intensity_profiles_smooth[:, :, channel]

    #extract number of pixels in the profiles and the number of angles used
    num_radial, num_angles = profile_image.shape

    #define in what region of the profile we are searching for membrane position.
    min_radius = min_radius_fraction * approx_radius
    max_radius = max_radius_fraction * approx_radius

    valid_radius = (
        (along_radius >= min_radius)
        & (along_radius <= max_radius)
    )

    radial_indices = np.where(valid_radius)[0]
    radius_values = along_radius[radial_indices]

    #if there are not enough pixels, discard GUV.
    if len(radial_indices) < 3:
        shape_x = np.full(num_angles, np.nan)
        shape_y = np.full(num_angles, np.nan)
        peak_positions = np.full(num_angles, np.nan)
        peak_found = np.full(num_angles, False)

        return shape_x, shape_y, peak_positions, peak_found

    #Extract intensities in the allowed region
    membrane_signal = profile_image[radial_indices, :]

    # Normalize intensity independently for each angle.
    #1. Extract min and max
    signal_min = np.nanmin(membrane_signal, axis=0, keepdims=True)
    signal_max = np.nanmax(membrane_signal, axis=0, keepdims=True)

    #2. minmax normalization
    signal_norm = (membrane_signal - signal_min) / (
        signal_max - signal_min + 1e-12
    )

    # create intensity cost: high signal should have low cost to be selected as membrane pixel.
    # it is done by changing the sign to the nornalized intnesity --> the more intense, the less
    # the cost to be selected.
    node_cost = -signal_norm

    # calculate distance from expected DisGUVery radius
    radius_prior = ((radius_values - approx_radius) / approx_radius) ** 2

    # Update coste by adding the cost of the distance from the expected radius. The further away from the
    # expected radius, the more cost. It is done by multiplaying the distance by the given radius distance weigth.
    node_cost = node_cost + radius_prior_weight * radius_prior[:, None]

    # Create a 2D matrix where the value at [i, j] is the cost of changing
    # from radius_values[i] at one angle to radius_values[j] at the next angle.
    # Small radius changes have low cost; large jumps have high cost.
    transition_cost = smoothness_weight * (
        radius_values[:, None] - radius_values[None, :]
    ) ** 2

    #calculates number of possible radial positons
    num_candidates = len(radius_values)

    #intiates 2 matrices the dimensions num of possible radial positiobs X num of angles used:
    #cummulative cost: stores cummulative cost for each allowed pixel for each angle
    cumulative_cost = np.zeros((num_candidates, num_angles))

    #backtrack: stores which previous candidate radius has been chosen.
    backtrack = np.zeros((num_candidates, num_angles), dtype=int)

    #since nothing has been chosen yet, no penalty for the distance to the previus pixel has been added.
    # So, it it the cost of the intensity + cost of distance to expected radius.
    cumulative_cost[:, 0] = node_cost[:, 0]

    #for each angle (starting from  the second one)
    for angle_i in range(1, num_angles):

        # adds the cost of the previus angle + cost of jumping from the previous position tot he new
        total_cost = (
            cumulative_cost[:, angle_i - 1][:, None]
            + transition_cost
        )

        # For each possible current radius, store pixel position with the lowest cost from the previous angle.
        best_previous_candidate = np.argmin(total_cost, axis=0)

        # Retrieve the corresponding minimum cost without searching twice.
        best_previous_cost = total_cost[
            best_previous_candidate,
            np.arange(num_candidates)
        ]

        backtrack[:, angle_i] = best_previous_candidate

        #store the best total cost for each candidate radius at the current angle.
        cumulative_cost[:, angle_i] = (
            node_cost[:, angle_i]
            + best_previous_cost
        )

    #crate an array to select one pixel per angle
    selected_candidate = np.zeros(num_angles, dtype=int)

    #Since we calculate commulative cost, the last angle candidate is the one with the lowest cummulative cost
    selected_candidate[-1] = np.argmin(cumulative_cost[:, -1])

    #select following candidates from the last to the first pixel (becasue, again, it is cummulative)
    # Recover the full best membrane path by moving backward through angles.
    for angle_i in range(num_angles - 2, -1, -1):

        next_angle = angle_i + 1
        next_candidate = selected_candidate[next_angle]

        selected_candidate[angle_i] = backtrack[next_candidate, next_angle]

    # converts selected indices into real radial positions.
    peak_positions = radius_values[selected_candidate]

    # smooth contour
    peak_positions = circular_smooth_peak_positions(
        peak_positions,
        window_size=final_smoothing_window
    )

    # Convert polar contour to XY coordinates
    shape_x = xc + peak_positions * np.cos(theta)
    shape_y = yc + peak_positions * np.sin(theta)

    peak_found = np.full(num_angles, True)

    return shape_x, shape_y, peak_positions, peak_found


def normalized_radial_profile_from_detected_shape(
    intensity_profiles,
    along_radius,
    peak_positions,
    num_channels,
    normalized_distance_axis=None
):
    """
    Calculate normalized radial profiles using angle-specific membrane positions.

    For each angular profile, the distance axis is normalized by the detected
    membrane position for that specific angle.

    Therefore, for every angle:
        detected membrane position = 1

    Input
    -----
    intensity_profiles : np.ndarray
        Intensity profiles with shape:
            radial position x angle/profile index x channel

    along_radius : np.ndarray
        Original distance values along each radial profile.

    peak_positions : np.ndarray
        Detected membrane radial position for each angle/profile.

    num_channels : int
        Number of image channels.

    normalized_distance_axis : np.ndarray or None
        Common normalized distance axis onto which profiles are interpolated.

    Returns
    -------
    radial_profiles_normalized : np.ndarray
        Average radial profiles after angle-specific distance normalization.
        Shape:
            normalized radial position x channel

    normalized_distance_axis : np.ndarray
        Normalized distance axis.

    normalized_profiles : np.ndarray
        Individual normalized profiles before angular averaging.
        Shape:
            normalized radial position x angle/profile index x channel
    """

    num_radial_points = intensity_profiles.shape[0]
    num_angles = intensity_profiles.shape[1]

    valid_peak_positions = peak_positions[
        ~np.isnan(peak_positions)
        & (peak_positions > 0)
    ]

    if len(valid_peak_positions) == 0:
        raise ValueError("No valid peak positions found for normalization.")

    if normalized_distance_axis is None:
        max_normalized_distance = np.nanmax(
            along_radius[-1] / valid_peak_positions
        )

        normalized_distance_axis = np.linspace(
            0,
            max_normalized_distance,
            num_radial_points
        )

    normalized_profiles = np.full(
        (
            len(normalized_distance_axis),
            num_angles,
            num_channels
        ),
        np.nan
    )

    for angle_i in range(num_angles):

        membrane_distance = peak_positions[angle_i]

        if np.isnan(membrane_distance) or membrane_distance <= 0:
            continue

        distance_normalized = along_radius / membrane_distance

        for ch in range(num_channels):
            normalized_profiles[:, angle_i, ch] = np.interp(
                normalized_distance_axis,
                distance_normalized,
                intensity_profiles[:, angle_i, ch],
                left=np.nan,
                right=np.nan
            )

    radial_profiles_normalized = np.nanmean(
        normalized_profiles,
        axis=1
    )

    return radial_profiles_normalized, normalized_distance_axis, normalized_profiles


def localization_shape_normalized(
    num_channels,
    normalized_profiles,
    radial_profiles_normalized,
    normalized_distance_axis,
    membrane_norm_in=0.95,
    membrane_norm_out=1.05,
    size_central_area=1/4,
    guv_channel=0
):
    """
    Quantify protein localization using shape-normalized profiles.

    Convention:
        normalized distance = 0 corresponds to the detected center
        normalized distance = 1 corresponds to the local detected membrane
        position for each angular profile.

    The localization is calculated for all channels except guv_channel.
    """

    comment = []

    protein_channels = [ch for ch in range(num_channels) if ch != guv_channel]

    membrane_mask = (
        (normalized_distance_axis >= membrane_norm_in)
        & (normalized_distance_axis <= membrane_norm_out)
    )

    centre_mask = normalized_distance_axis <= size_central_area

    if not np.any(membrane_mask):
        raise ValueError("No membrane region selected on normalized axis.")

    if not np.any(centre_mask):
        raise ValueError("No centre region selected on normalized axis.")

    angular_profiles_normalized = np.nanmean(
        normalized_profiles[membrane_mask, :, :],
        axis=0
    )

    localization_index = np.zeros(len(protein_channels))

    for protein_i, ch in enumerate(protein_channels):

        mean_angular_signal = np.nanmean(angular_profiles_normalized[:, ch])

        if mean_angular_signal > 0:
            rsd = np.nanstd(angular_profiles_normalized[:, ch]) / mean_angular_signal
        else:
            rsd = np.nan
            comment = comment + [f"zero_mean_ch{ch}"]

        if not np.isnan(rsd) and rsd > 0.8:
            comment = comment + [f"high_rsd_ch{ch}"]

        membrane = np.nanmedian(angular_profiles_normalized[:, ch])
        centre = np.nanmean(radial_profiles_normalized[centre_mask, ch])

        if membrane > 0:
            localization_index[protein_i] = np.clip(
                (membrane - centre) / membrane,
                0,
                None
            )
        else:
            localization_index[protein_i] = 0
            comment = comment + [f"zero_median_ch{ch}"]

    return localization_index, protein_channels, comment, angular_profiles_normalized


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

    crop = zoom_in_vesicle(
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

        crop = zoom_in_vesicle(
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

def save_membrane_positions(
    output_folder,
    image_path,
    membrane_positions
):
    """
    Save detected membrane positions for all GUVs in one source image.

    The JSON contains one entry per vesicle id.
    Each vesicle contains a list of [x, y] membrane coordinates.
    """

    image_name = os.path.basename(image_path)
    image_stem = os.path.splitext(image_name)[0]

    membrane_data = {
        "source_file": image_name,
        "vesicles": membrane_positions
    }

    membrane_path = os.path.join(
        output_folder,
        f"{image_stem}_membrane_positions.json"
    )

    with open(membrane_path, "w") as f:
        json.dump(membrane_data, f, indent=4)

    return membrane_path


def load_membrane_positions_json(membrane_path):
    """
    Load detected membrane positions from one JSON file.

    Returns
    -------
    source_file : str
        Name of the original image file.

    vesicles : dict
        Dictionary where each key is a vesicle id and each value is a
        NumPy array with shape:
            num_angles x 2

        Column 0 is x.
        Column 1 is y.
    """

    with open(membrane_path, "r") as f:
        membrane_data = json.load(f)

    vesicles = {
        vesicle_id: np.array(membrane_xy)
        for vesicle_id, membrane_xy in membrane_data["vesicles"].items()
    }

    return membrane_data["source_file"], vesicles

def save_shape_normalized_profiles(
    output_folder,
    image_path,
    profiles_data
):
    """
    Save shape-normalized radial and angular profiles for all GUVs in one source image.

    The NPZ file contains:
        radial profile per vesicle
        angular profile per vesicle
        normalized distance axis per vesicle
    """

    image_name = os.path.basename(image_path)
    image_stem = os.path.splitext(image_name)[0]

    profile_path = os.path.join(
        output_folder,
        f"{image_stem}_shape_normalized_profiles.npz"
    )

    np.savez_compressed(profile_path, **profiles_data)

    return profile_path

def save_membrane_positions(
    output_folder,
    image_path,
    membrane_positions
):
    """
    Save detected membrane positions for all GUVs in one source image.

    The JSON contains one entry per vesicle id.
    Each vesicle contains a list of [x, y] membrane coordinates.
    """

    image_name = os.path.basename(image_path)
    image_stem = os.path.splitext(image_name)[0]

    membrane_data = {
        "source_file": image_name,
        "vesicles": membrane_positions
    }

    membrane_path = os.path.join(
        output_folder,
        f"{image_stem}_membrane_positions.json"
    )

    with open(membrane_path, "w") as f:
        json.dump(membrane_data, f, indent=4)

    return membrane_path

def reorder_summary_columns(results_df):
    """
    Reorder summary table columns so paths and main identifiers appear first.
    """

    first_columns = [
        "image_path",
        "membrane_position_file",
        "profile_data_file",
        "vesicle_id",
        "xc",
        "yc",
        "radius",
    ]

    first_columns = [
        col for col in first_columns
        if col in results_df.columns
    ]

    other_columns = [
        col for col in results_df.columns
        if col not in first_columns
    ]

    results_df = results_df[
        first_columns + other_columns
    ]

    return results_df
# %% 1. Detect membrane positions and save JSON

images = glob(os.path.join(PATH, "**", "*"+IMAGE_FORMAT), recursive=True)

detection_results = []

for image_i, image_path in tqdm(enumerate(images), total=len(images)):
    print(image_path)
    output_folder = get_output_folder(image_path, PATH)
    membrane_positions = {}

    img = open_image(image_path)

    csv_path = image_path.replace(IMAGE_FORMAT, DETECTION_SUFFIX)

    if not os.path.exists(csv_path):
        print(f"Skipping image. Detection CSV not found: {csv_path}")
        continue

    locs = pd.read_csv(csv_path, index_col=False).to_numpy()

    num_vesicles = len(locs[:, 0])
    num_channels = img.shape[0]

    image_dim = np.array((img.shape[2], img.shape[1]))  # X, Y

    image_results = []

    for ves_i in range(num_vesicles):

        comment = []

        ves_coordinates = locs[ves_i, :]
        vesicle_id = int(ves_coordinates[0])

        # 1. Calculate raw linear profiles from original detected centre
        intensity_profiles, along_radius, theta, death_mark = linear_profiles(
            img,
            ves_coordinates,
            image_dim,
            PARAMETERS_PROFILES
        )

        along_radius_ori = along_radius.copy()

        # 2. Skip vesicles too close to the image border
        if death_mark:

            comment = ["margins"]

            image_results.append({
                "image_path": image_path,
                "vesicle_id": vesicle_id,
                "xc": ves_coordinates[1],
                "yc": ves_coordinates[2],
                "radius": ves_coordinates[3],
                "death_mark": True,
                "death_mark_shape": False,
                "valid_shape_fraction": 0,
                "comment": ";".join(comment),
            })

            print("Skipped GUV: profiles extend outside image margins.")
            continue

        # 3. Smooth neighboring angular profiles
        intensity_profiles_smooth = circular_rolling_average_profiles(
            intensity_profiles,
            window_size=SMOOTH_WINDOW
        )

        # 4. Estimate local background using annulus around vesicle
        background = background_noise(
            img,
            ves_coordinates,
            inner_margin=INNER_MARGIN,
            outer_margin=OUTER_MARGIN
        )

        # 5. Background correction
        intensity_profiles_smooth_corrected = background_correction(
            num_channels,
            intensity_profiles_smooth,
            background
        )

        # 6. Detect membrane shape from GUV channel using global smooth path
        shape_x, shape_y, peak_positions, peak_found = detect_membrane_shape_dp(
            intensity_profiles_smooth=intensity_profiles_smooth_corrected,
            along_radius=along_radius_ori,
            theta=theta,
            ves_coordinates=ves_coordinates,
            channel=GUV_CH,
            smoothness_weight=DP_SMOOTHNESS_WEIGHT,
            radius_prior_weight=DP_RADIUS_PRIOR_WEIGHT,
            min_radius_fraction=DP_MIN_RADIUS_FRACTION,
            max_radius_fraction=DP_MAX_RADIUS_FRACTION,
            final_smoothing_window=DP_FINAL_SMOOTHING_WINDOW
        )

        valid_shape = ~np.isnan(shape_x) & ~np.isnan(shape_y)
        valid_shape_fraction = np.mean(valid_shape)

        if valid_shape_fraction < MIN_VALID_SHAPE_FRACTION:

            comment = comment + ["poor_shape_detection"]

            image_results.append({
                "image_path": image_path,
                "vesicle_id": vesicle_id,
                "xc": ves_coordinates[1],
                "yc": ves_coordinates[2],
                "radius": ves_coordinates[3],
                "death_mark": True,
                "death_mark_shape": True,
                "valid_shape_fraction": valid_shape_fraction,
                "comment": ";".join(comment),
            })

            print("Skipped GUV: poor shape detection.")
            continue

        # Store membrane position as list of [x, y] points
        membrane_positions[str(vesicle_id)] = np.column_stack(
            (shape_x, shape_y)
        ).tolist()

        image_results.append({
            "image_path": image_path,
            "vesicle_id": vesicle_id,
            "xc": ves_coordinates[1],
            "yc": ves_coordinates[2],
            "radius": ves_coordinates[3],
            "death_mark": False,
            "death_mark_shape": False,
            "valid_shape_fraction": valid_shape_fraction,
            "comment": ";".join(comment),
        })

        # Optional visual check of detected shape
        image_stem = get_image_stem(image_path)

        shape_plot_path = os.path.join(
            output_folder,
            f"{image_stem}_ves{vesicle_id}_membrane_detection.png"
        )

        plot_detected_guv_shape(
            channel_data=img[GUV_CH, :, :],
            ves_coordinates=ves_coordinates,
            image_dim=image_dim,
            shape_x=shape_x,
            shape_y=shape_y,
            size_view=SIZE_VIEW,
            title=f"Detected GUV shape - vesicle {vesicle_id}",
            save_path=shape_plot_path,
            show=PLOT_RESULTS
        )

    membrane_path = save_membrane_positions(
        output_folder=output_folder,
        image_path=image_path,
        membrane_positions=membrane_positions
    )

    for result_row in image_results:
        result_row["membrane_position_file"] = membrane_path
        detection_results.append(result_row)


detection_results_df = pd.DataFrame(detection_results)

detection_results_path = os.path.join(PATH, "membrane_detection_results.csv")
detection_results_df.to_csv(detection_results_path, index=False)

print(f"\nDone. Membrane detection results saved to: {detection_results_path}")


# %% EXAMPLE Load membrane JSON and plot GUV outline

membrane_path = r"D:\Data\EVOLF\output\20260617_GUV_FLuo4_TRIMEB_aHL_perm_2_test_analysis_radial_profile\20260610_D4_t0-04_membrane_positions.json"

source_file, vesicles = load_membrane_positions_json(membrane_path)

print(source_file)
print(vesicles.keys())

vesicle_id = "1"

membrane_xy = vesicles[vesicle_id]

membrane_xy_closed = np.vstack([
    membrane_xy,
    membrane_xy[0, :]
])

plt.figure()
plt.plot(membrane_xy_closed[:, 0], membrane_xy_closed[:, 1], ".-")
plt.axis("equal")
plt.xlabel("x")
plt.ylabel("y")
plt.title(f"GUV outline - vesicle {vesicle_id}")
plt.show()


# %% 2. Shape-normalized radial profile and plotting for all saved membrane JSON files

# Find all saved membrane-position JSON files
membrane_jsons = glob(
    os.path.join(PATH, "output", "**", "*membrane_positions.json"),
    recursive=True
)

# Find all original CZI files once
images = glob(os.path.join(PATH, "**", "*"+IMAGE_FORMAT), recursive=True)

# Store results from all images and all GUVs in one list
profile_results = []

for membrane_path in tqdm(membrane_jsons):

    print(f"\nProcessing membrane file: {membrane_path}")

    source_file, vesicles = load_membrane_positions_json(membrane_path)
    image_profile_results = []
    profiles_data = {}

    image_name = source_file
    image_path = None

    for candidate_image_path in images:
        if os.path.basename(candidate_image_path) == image_name:
            image_path = candidate_image_path
            break

    if image_path is None:
        print(f"Skipping. Could not find source image: {image_name}")
        continue

    print(image_path)

    output_folder = get_output_folder(image_path, PATH)

    img = open_image(image_path)

    csv_path = image_path.replace(IMAGE_FORMAT, DETECTION_SUFFIX)

    if not os.path.exists(csv_path):
        print(f"Skipping image. Detection CSV not found: {csv_path}")
        continue

    locs = pd.read_csv(csv_path, index_col=False).to_numpy()

    num_channels = img.shape[0]

    image_dim = np.array((img.shape[2], img.shape[1]))  # X, Y

    for vesicle_id, membrane_xy in vesicles.items():

        vesicle_id_int = int(vesicle_id)

        matching_rows = locs[locs[:, 0].astype(int) == vesicle_id_int]

        if len(matching_rows) == 0:
            print(f"Skipping vesicle {vesicle_id}: not found in detection CSV.")
            continue

        ves_coordinates = matching_rows[0, :]

        # 1. Calculate raw linear profiles from original detected centre
        intensity_profiles, along_radius, theta, death_mark = linear_profiles(
            img,
            ves_coordinates,
            image_dim,
            PARAMETERS_PROFILES
        )

        along_radius_ori = along_radius.copy()

        if death_mark:
            print(f"Skipping vesicle {vesicle_id}: profiles extend outside image margins.")
            continue

        # 2. Smooth neighboring angular profiles
        intensity_profiles_smooth = circular_rolling_average_profiles(
            intensity_profiles,
            window_size=SMOOTH_WINDOW
        )

        # 3. Estimate local background using annulus around vesicle
        background = background_noise(
            img,
            ves_coordinates,
            inner_margin=INNER_MARGIN,
            outer_margin=OUTER_MARGIN
        )

        # 4. Background correction
        intensity_profiles_smooth_corrected = background_correction(
            num_channels,
            intensity_profiles_smooth,
            background
        )

        # Convert saved XY membrane contour back to radial membrane distance
        xc = ves_coordinates[1]
        yc = ves_coordinates[2]

        peak_positions = np.sqrt(
            (membrane_xy[:, 0] - xc) ** 2
            + (membrane_xy[:, 1] - yc) ** 2
        )

        # 5. Create normalized radial profile using local membrane distance
        radial_profiles_normalized, normalized_distance_axis, normalized_profiles = (
            normalized_radial_profile_from_detected_shape(
                intensity_profiles=intensity_profiles_smooth_corrected,
                along_radius=along_radius_ori,
                peak_positions=peak_positions,
                num_channels=num_channels
            )
        )

        # 6. Shape-normalized localization
        localization_index, localization_channels, comment_loc, angular_profiles_normalized = (
            localization_shape_normalized(
                num_channels=num_channels,
                normalized_profiles=normalized_profiles,
                radial_profiles_normalized=radial_profiles_normalized,
                normalized_distance_axis=normalized_distance_axis,
                membrane_norm_in=MEMBRANE_NORM_IN,
                membrane_norm_out=MEMBRANE_NORM_OUT,
                size_central_area=size_central_area,
                guv_channel=GUV_CH
            )
        )

        profiles_data[f"ves{vesicle_id}_radial_profiles_normalized"] = radial_profiles_normalized
        profiles_data[f"ves{vesicle_id}_normalized_distance_axis"] = normalized_distance_axis
        profiles_data[f"ves{vesicle_id}_angular_profiles_normalized"] = angular_profiles_normalized

        image_stem = get_image_stem(image_path)

        combined_plot_path = os.path.join(
            output_folder,
            f"{image_stem}_ves{vesicle_id}_shape_normalized_profiles.png"
        )

        plot_shape_normalized_profiles_crops_and_angles(
            channels_data=img,
            ves_coordinates=ves_coordinates,
            image_dim=image_dim,
            radial_profiles_normalized=radial_profiles_normalized,
            normalized_distance_axis=normalized_distance_axis,
            angular_profiles_normalized=angular_profiles_normalized,
            channels=tuple(range(num_channels)),
            size_view=SIZE_VIEW,
            title=f"Shape-normalized profiles - vesicle {vesicle_id}",
            save_path=combined_plot_path,
            show=PLOT_RESULTS
        )

        result_row = {
            "image_path": image_path,
            "membrane_position_file": membrane_path,
            "vesicle_id": vesicle_id_int,
            "xc": ves_coordinates[1],
            "yc": ves_coordinates[2],
            "radius": ves_coordinates[3],
            "comment": ";".join(comment_loc),
        }

        for ch in range(num_channels):
            result_row[f"background_ch{ch}"] = background[0, ch]

        for loc_value, ch in zip(localization_index, localization_channels):
            result_row[f"localization_ch{ch}"] = loc_value

        image_profile_results.append(result_row)
    
    profile_data_path = save_shape_normalized_profiles(
        output_folder=output_folder,
        image_path=image_path,
        profiles_data=profiles_data
    )

    for result_row in image_profile_results:
        result_row["profile_data_file"] = profile_data_path
        profile_results.append(result_row)


# Save one CSV with all GUVs from all images
profile_results_df = pd.DataFrame(profile_results)
profile_results_df = reorder_summary_columns(profile_results_df)

profile_results_path = os.path.join(PATH, "shape_normalized_profile_results.csv")
profile_results_df.to_csv(profile_results_path, index=False)

print(f"\nDone. Shape-normalized profile results saved to: {profile_results_path}")