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
from scipy.interpolate import interp1d


def linear_profiles(channels_data, ves_coordinates, along_radius, theta, parameters_profiles):
    """
    Calculation of the multiple linear profiles for a single vesicle.

    Input
    -----
    channels_data : np.ndarray
        The channels of the image to be analyzed.
        channels_data[0, :, :] always corresponds to the membrane channel.

    ves_coordinates : np.ndarray
        1-dimensional array with the coordinates of the single vesicle we want
        to focus on.
        ves_coordinates[0]: vesicle id
        ves_coordinates[1]: xc (in px)
        ves_coordinates[2]: yc (in px)
        ves_coordinates[3]: radius (in px)

    parameters_profiles : np.ndarray
        The 3 important parameters for the linear profile calculation:
        parameters_profiles[0]: number of linear profiles to consider for a
                                single vesicle
        parameters_profiles[1]: length of the linear profiles
                                (unit of measure: vesicle radius)
        parameters_profiles[2]: the step (in px) for sampling along the vesicle
                                radius

    Returns
    -------
    intensity_profiles : np.ndarray
        The intensity linear profiles calculated.

        Data related to different channels are stored along the 3rd dimension
        of the array.

        Element[:, :, 0] is always based on the membrane channel.
        Element[:, :, 1] is based on the first protein channel.
        If both proteins are present, element[:, :, 1] corresponds to the
        septin channel and element[:, :, 2] corresponds to the actin channel.

    along_radius : np.ndarray
        The points along the linear profiles, starting from the center of the
        vesicle.

    theta : np.ndarray
        The angles considered for the linear profiles, in radians.

    death_mark : bool
        If True, the requested intensity profile extends outside the image
        borders and the vesicle should be disregarded.

        If False, the vesicle position does not raise an issue and the analysis
        can continue.
    """

    num_channels = channels_data.shape[0]

    image_dim = np.array((channels_data.shape[2], channels_data.shape[1]))

    xc = ves_coordinates[1]
    yc = ves_coordinates[2]
    radius = ves_coordinates[3]


    profile_radius_limit = parameters_profiles.length_excess * radius

    profile_radius = len(along_radius)

    # Create an empty array with the expected output shape.
    num_angles = len(theta)
    intensity_profiles = np.zeros((profile_radius, num_angles, num_channels))

    death_mark = False

    # Check whether any radial profile would extend outside the image.
    if (
        xc + profile_radius_limit >= image_dim[0]
        or xc - profile_radius_limit < 0
        or yc + profile_radius_limit >= image_dim[1]
        or yc - profile_radius_limit < 0
    ):
        death_mark = True

        return intensity_profiles, death_mark, ['too close to the border']

    # Convert the radial positions into a column array.
    # Shape: (profile_radius, 1)
    radial_positions = along_radius[:, np.newaxis]

    # Convert the angles into a row array.
    # Shape: (1, num_angles)
    angles = theta[np.newaxis, :]

    # Create the x-coordinate of every sampling point along each radial profile.
    # Each row corresponds to a distance from the center.
    # Each column corresponds to one angle.
    # x = xc + radius_position * cos(angle)
    line_x = (xc + np.rint(
                radial_positions
                * np.cos(angles)
            )
        ).astype(np.intp)

    # Create the y-coordinate of every sampling point along each radial profile:
    # y = yc + radius_position * sin(angle)
    line_y = (yc + np.rint(
                radial_positions
                * np.sin(angles)
            )
        ).astype(np.intp)

    # Extract the intensity values for all channels and all profile coordinates.
    intensity_profiles = channels_data[:, line_y, line_x]

    # Move the channel dimension from the first position to the last position.
    intensity_profiles = np.moveaxis(intensity_profiles,  0, -1).astype(np.float64)

    return intensity_profiles, death_mark, []


def trim_central_profiles(intensity_profiles, pixels_to_remove,):
    """
    Remove unreliable radial samples near the vesicle center.

    The innermost samples are excluded because coordinate rounding can cause
    multiple angular profiles to sample the same image pixels.

    Parameters
    ----------
    intensity_profiles : np.ndarray
        Intensity profiles with shape
        ``(radial_positions, angles, channels)``.

    along_radius : np.ndarray
        Radial coordinates corresponding to axis 0 of
        ``intensity_profiles``.

    pixels_to_remove : int
        Number of samples to remove from the beginning of the radial axis.

    Returns
    -------
    intensity_profiles : np.ndarray
        Trimmed intensity profiles.

    along_radius : np.ndarray
        Radial coordinates corresponding to the trimmed profiles.
    """
    return intensity_profiles[pixels_to_remove:, :, :]

def circular_rolling_average_linear_profiles(intensity_profiles, window_size=5):
    """
    Smooth intensity profiles by circular rolling average of the intensity profiles along the angle axis.

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

def radial_profile(intensity_profiles):
    """
    Average the angular intensity profiles into one radial profile per channel.

    Parameters
    ----------
    intensity_profiles : np.ndarray
        Array with shape (radial_positions, angles, channels).

    Returns
    -------
    np.ndarray
        Mean radial profile with shape (radial_positions, channels).
    """
# TODO: Add optional normalization and interpolation onto a shared radial axis in the wrapper function in the class.
# For circular GUVs, first calculate the mean radial profile, normalize
# along_radius by the GUV radius, and interpolate all channels onto a fixed
# axis such as np.linspace(0, max_normalized_distance, num_normalized_points).
# This ensures profiles from GUVs with different radii have the same length
# and can be compared, stacked, or averaged directly.

    return np.mean(intensity_profiles, axis=1)


def normalized_radial_profile_from_detected_shape(intensity_profiles, along_radius, peak_positions, max_normalized_distance=1.5, num_normalized_points=100):
    """
    Calculate normalized radial profiles using angle-specific membrane positions.

    For each angular profile, the distance axis is normalized by the detected
    membrane position for that specific angle.

    Therefore, for every angle:
        detected membrane position = 1

    Input
    ----------
    intensity_profiles : np.ndarray
        Intensity profiles with shape
        ``(radial_positions, angles, channels)``.

    along_radius : np.ndarray
        One-dimensional radial distance axis corresponding to the first
        dimension of ``intensity_profiles``.

    peak_positions : np.ndarray
        Detected membrane position for each angle. Must have one value per
        angular profile. Missing detections may be represented by ``np.nan``.

    max_normalized_distance : float, default=1.5
        Maximum value of the normalized radial axis.

        For example, a value of 1.5 creates an axis extending from the GUV
        center at 0 to 1.5 times the angle-specific membrane distance.
        The membrane itself is located at 1.

    num_normalized_points : int, default=100
        Number of points in the common normalized radial axis.

    Output
    -------
    radial_profiles_normalized : np.ndarray
        Mean normalized radial profile, averaged over valid angles.
        Shape: ``(num_normalized_points, channels)``.

    normalized_distance_axis : np.ndarray
        Common normalized radial axis.
        Shape: ``(num_normalized_points,)``.

    normalized_profiles : np.ndarray
        Individual normalized profiles before averaging over angles.
        Shape: ``(num_normalized_points, angles, channels)``.
    """
    _, num_angles, num_channels = intensity_profiles.shape

    valid_peak_positions = (~np.isnan(peak_positions) & (peak_positions > 0))

    if not np.any(valid_peak_positions):
        raise ValueError("No valid peak positions found for normalization.")

    # Shared axis used for every normalized angular profile
    normalized_distance_axis = np.linspace(0, max_normalized_distance, num_normalized_points)

    normalized_profiles = np.full((num_normalized_points, num_angles, num_channels), np.nan, dtype=float)

    for angle_i in range(num_angles):
        membrane_distance = peak_positions[angle_i]

        # Skip angles without a valid membrane detection
        if np.isnan(membrane_distance) or membrane_distance <= 0:
            continue

        # Set the detected membrane position to normalized distance 1
        distance_normalized = along_radius / membrane_distance

        # Interpolate all channels for this angle at once
        interpolator = interp1d(
            distance_normalized,
            intensity_profiles[:, angle_i, :],
            axis=0,
            bounds_error=False,
            fill_value=np.nan,
        )

        normalized_profiles[:, angle_i, :] = interpolator(normalized_distance_axis)

    # Average the aligned profiles over angle
    radial_profiles_normalized = np.nanmean(normalized_profiles, axis=1)

    return radial_profiles_normalized, normalized_distance_axis, normalized_profiles
    

def angular_profile(intensity_profiles, index_border_in, index_border_out, settings, guv_ch):
    """
    Calculation of the angular profile along the membrane contour for a given 
    vesicle.
                        
    Input
    -----
    intensity_profiles : np.ndarray
        The intensity linear profiles calculated. Backgrounds corrections should
        take place beforehand. 
        Note: Data related to different channels are stored along the 3rd 
              dimension of the array. Element [:,:,i] refers tyo each channel.
              
    index_border_in : int
        The index of the element corresponding to the inner border of the membrane.
        
    index_border_out : int
        The index of the element corresponding to the outer border of the membrane.
        
    Returns
    -------
    angular_profiles : np.ndarray
        The angular profile along the membrane contour for a given vesicle.
        The profiles corresponding to the different channels are stored along 
        the 3rd dimension. 
        
    
    """

    start = index_border_in
    stop = index_border_out

    num_angles = intensity_profiles.shape[1]
    theta = np.linspace(0, 2 * np.pi, num_angles, endpoint=False)

    angular_profiles = np.mean(intensity_profiles[start:stop, :, :], axis=0)
    if settings.flatten_angular_profiles:
        flattened_profile, fitted_profile, death_mark, comments = flatten_angular_profile(angular_profiles[:,guv_ch], theta, settings)
    
    return angular_profiles, flattened_profile, death_mark, comments



def angular_profile_from_detected_shape(intensity_profiles, along_radius, peak_radius_by_angle, settings, guv_ch):
    """
    Calculate angular intensity profiles around a detected noncircular membrane.

    Parameters
    ----------
    intensity_profiles : np.ndarray
        Intensity profiles with shape radial position x angle x channel.

    along_radius : np.ndarray
        Radial distance corresponding to the first axis of intensity_profiles.

    peak_radius_by_angle : np.ndarray
        Detected membrane radius for each angle.

    membrane_width_pixels : int
        Odd number of radial samples averaged around each detected membrane position.
        For example, 5 uses the central membrane sample and 2 samples on each side.

    Returns
    -------
    angular_profiles : np.ndarray
        Mean membrane intensity with shape angle x channel.
    """
    if settings.noncircular_membrane_width_pixels < 1 or settings.noncircular_membrane_width_pixels % 2 == 0:
        raise ValueError("membrane_width_pixels must be a positive odd number.")

    _, num_angles, num_channels = intensity_profiles.shape

    if peak_radius_by_angle.size != num_angles:
        raise ValueError("peak_radius_by_angle must contain one radius per angle.")

    half_width = settings.noncircular_membrane_width_pixels // 2
    angular_profiles = np.full((num_angles, num_channels), np.nan)

    for angle_i, peak_radius in enumerate(peak_radius_by_angle):
        if np.isnan(peak_radius):
            continue

        #find the radial sample closest to the detected membrane radius
        peak_index = np.argmin(np.abs(along_radius - peak_radius))

        #select an equal number of radial samples inside and outside the membrane
        start_index = max(0, peak_index - half_width)
        end_index = min(along_radius.size, peak_index + half_width + 1)

        #skip angles where the complete requested window does not fit
        if end_index - start_index != settings.noncircular_membrane_width_pixels:
            continue

        #average the selected membrane region independently for every channel
        angular_profiles[angle_i, :] = np.nanmean(intensity_profiles[start_index:end_index, angle_i, :], axis=0)

    num_angles = intensity_profiles.shape[1]
    theta = np.linspace(0, 2 * np.pi, num_angles, endpoint=False)
    if settings.flatten_angular_profiles:
        flattened_profile, fitted_profile, death_mark, comments = flatten_angular_profile(angular_profiles[:,guv_ch], theta, settings)
    
    return angular_profiles, flattened_profile, death_mark, comments

def flatten_angular_profile(angular_profile, angles, settings):
    """
    Remove the two-fold angular intensity modulation caused by polarization.

    Parameters
    ----------
    angular_profile : np.ndarray
        Membrane intensity at each angle.

    angles : np.ndarray
        Angular positions in radians.

    Returns
    -------
    flattened_profile : np.ndarray
        Angular intensity corrected for the fitted polarization pattern and
        rescaled to the median intensity of the original angular profile.

    fitted_profile : np.ndarray
        Fitted two-fold angular intensity pattern.
    """
    comments = []
    death_mark = False
    valid = np.isfinite(angular_profile) & np.isfinite(angles)

    design_matrix = np.column_stack((
        np.ones(np.count_nonzero(valid)),
        np.cos(2 * angles[valid]),
        np.sin(2 * angles[valid]),
    ))

    profile = angular_profile[valid]

    coefficients, _, _, _ = np.linalg.lstsq(design_matrix, profile, rcond=None)

    full_design_matrix = np.column_stack((
        np.ones(angles.size),
        np.cos(2 * angles),
        np.sin(2 * angles),
    ))

    fitted_profile = full_design_matrix @ coefficients

    if np.any(~np.isfinite(fitted_profile)) or np.any(fitted_profile <= 0):
        comments.append("invalid_polarization_fit")
        return angular_profile.copy(), fitted_profile, False, comments

    flattened_profile = angular_profile / fitted_profile

    fit = fitted_profile[valid]

    profile_median = np.nanmedian(profile)
    fit_median = np.nanmedian(fit)

    if not np.isfinite(profile_median) or profile_median <= 0:
        raise ValueError("Angular profile has a non-positive or non-finite median.")

    if not np.isfinite(fit_median) or fit_median <= 0:
        raise ValueError("Polarization fit has a non-positive or non-finite median.")

    profile_norm = profile / profile_median
    fit_norm = fit / fit_median

    fit_rmse = np.sqrt(np.nanmean((profile_norm - fit_norm) ** 2))
    fit_correlation = np.corrcoef(profile_norm, fit_norm)[0, 1]

    a, b, c = coefficients

    if not np.isfinite(a) or a <= 0:
        raise ValueError("Polarization fit has a non-positive or non-finite baseline.")

    amplitude = np.hypot(b, c)
    modulation_depth = amplitude / a
    std = np.std(flattened_profile)

    if modulation_depth < settings.min_angular_variation:
        comments.append("angular_intensity_variation_too_low")

    if fit_rmse > settings.max_angular_fit_error:
        comments.append("angular_intensity_fit_error_too_high")

    if not np.isfinite(fit_correlation) or fit_correlation < settings.min_angular_fit_correlation:
        comments.append("angular_intensity_fit_correlation_too_low")

    if std > 0.4: 
        comments.append("high angular error")
    
    flattened_profile *= np.nanmedian(angular_profile)

    return flattened_profile, fitted_profile, death_mark, comments

def circular_rolling_average(profile, window=9):
    """Calculate a centered rolling average for a circular one-dimensional profile."""
    profile = np.asarray(profile, dtype=float)

    if window < 1 or window % 2 == 0:
        raise ValueError("window must be a positive odd number.")

    half_window = window // 2
    padded_profile = np.pad(profile, half_window, mode="wrap")
    kernel = np.ones(window) / window
    return np.convolve(padded_profile, kernel, mode="valid")

def choose_num_angles(radius, target_arc_spacing=1.1, min_angles=120, max_angles=360):
    """
    Choose the number of angular profiles based on vesicle size.

    The goal is to keep the spacing between neighboring angular profiles
    approximately constant at the membrane.

    Math
    ----
    If the vesicle is approximated as a circle with radius r, then the
    membrane length is the circumference:

        circumference = 2 * pi * r

    If we extract N angular profiles around the vesicle, then the approximate
    distance along the membrane between two neighboring profiles is:

        arc_spacing = circumference / N

    Therefore:

        arc_spacing = 2 * pi * r / N

    Here, we choose the desired arc spacing first. This is called
    target_arc_spacing. Then we solve for N:

        N = 2 * pi * r / target_arc_spacing

    For example, if:

        radius = 40 pixels
        target_arc_spacing = 1.1 pixels

    then:

        N = 2 * pi * 40 / 1.1
        N ≈ 229 angles

    Smaller target_arc_spacing gives more angular profiles and more detail.
    Larger target_arc_spacing gives fewer angular profiles and faster analysis.

    Input
    -----
    radius : float
        Approximate vesicle radius in pixels.

    target_arc_spacing : float
        Desired approximate spacing in pixels between neighboring angular
        profiles at the membrane.

    min_angles : int
        Minimum allowed number of angular profiles.

    max_angles : int
        Maximum allowed number of angular profiles.

    Returns
    -------
    num_angles : int
        Number of angular profiles to use for this vesicle.
    """

    num_angles = int(np.ceil(2 * np.pi * radius / target_arc_spacing))

    num_angles = np.clip(num_angles, min_angles, max_angles)

    return int(num_angles)