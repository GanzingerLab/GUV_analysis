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
from scipy.ndimage import uniform_filter1d
from .profiles import circular_rolling_average_linear_profiles

def detect_circular_GUV(radial_profile_memb, along_radius, radius, settings):
    """
    Detect the circular GUV membrane peak and its inner and outer borders.

    The radial intensity profile is normalized before peak detection. When multiple
    peaks are found, the outermost peak is selected initially, but a stronger nearby
    inner peak can replace it when the outer peak lies beyond the approximate GUV radius.

    Parameters
    ----------
    radial_profile_memb : np.ndarray
        One-dimensional radial intensity profile of the membrane channel.

    along_radius : np.ndarray
        Radial distance corresponding to each element of `radial_profile_memb`.
        It must have the same length as the radial profile.

    radius : float
        Approximate GUV radius in the same units as `along_radius`.

    settings : CircularMembraneSettings
        Settings controlling peak detection, peak-width measurement and membrane
        quality checks.

    Returns
    -------
    peak_radius : float
        Radial distance of the selected membrane peak.

    index_border_in : int
        Array index of the inner membrane border.

    index_border_out : int
        Array index of the outer membrane border.

    comments : list[str]
        Quality-control comments generated during membrane detection.

    death_mark : bool
        True when no membrane peak could be detected.
    """

    comments = []

    #normalize the radial profile between 0 and 1
    valid = np.isfinite(radial_profile_memb)

    if not np.any(valid):
        return np.nan, 0, 0, 0, ["no_valid_radial_profile"], True

    radial_profile_memb = radial_profile_memb.copy()
    radial_profile_memb[~valid] = np.nanmin(radial_profile_memb)

    max_intensity = np.nanmax(radial_profile_memb)
    min_intensity = np.nanmin(radial_profile_memb)

    if max_intensity == min_intensity:
        norm_radial_profile = np.zeros_like(radial_profile_memb, dtype=float)
    else:
        norm_radial_profile = (radial_profile_memb - min_intensity) / (max_intensity - min_intensity)

    #detect candidate membrane peaks in the normalized radial profile
    peaks, _ = find_peaks(norm_radial_profile, height=settings.peak_height, distance=settings.peak_distance, prominence=settings.peak_prominence)

    #discard the GUV when no membrane peak is found
    if peaks.size == 0:
        return np.nan, 0, 0, 0, ["no_memb_peak"], True

    #measure the width of each candidate peak at the configured relative height
    widths = peak_widths(radial_profile_memb, peaks, rel_height=settings.width_relative_height)

    #initially select the outermost detected peak
    chosen_peak = len(peaks) - 1

    #compare the outermost peak with up to two neighboring inner peaks
    if len(peaks) > 1:
        candidate_indices = range(chosen_peak - 1, max(-1, chosen_peak - 3), -1)

        for candidate in candidate_indices:
            chosen_index = peaks[chosen_peak]
            candidate_index = peaks[candidate]
            chosen_radius = along_radius[chosen_index]
            chosen_height = radial_profile_memb[chosen_index]
            candidate_height = radial_profile_memb[candidate_index]

            #replace a weak peak outside the approximate radius with a stronger inner peak
            if chosen_radius > radius and chosen_height < candidate_height:
                chosen_peak = candidate

        #a selected peak other than the innermost peak indicates several internal peaks
        if chosen_peak > 0:
            comments.append("confetti")

    #extract the selected peak index, physical radius and intensity
    peak_index = peaks[chosen_peak]
    peak_radius = interpolate_peak_radius(norm_radial_profile, along_radius, peak_index)
    peak_height = radial_profile_memb[peak_index]

    #extract the fractional inner and outer border positions returned by peak_widths
    border_in_position = widths[2][chosen_peak]
    border_out_position = widths[3][chosen_peak]

    #convert the fractional border positions to radial distances for the width check
    profile_indices = np.arange(radial_profile_memb.size)
    border_in_radius = np.interp(border_in_position, profile_indices, along_radius)
    border_out_radius = np.interp(border_out_position, profile_indices, along_radius)
    membrane_width = border_out_radius - border_in_radius

    #convert the border positions to valid array indices for later profile indexing
    index_border_in = int(np.clip(np.rint(border_in_position), 0, radial_profile_memb.size - 1))
    index_border_out = int(np.clip(np.rint(border_out_position), 0, radial_profile_memb.size))

    #flag a membrane peak that is too wide relative to the approximate GUV radius
    if membrane_width > radius * settings.wide_peak_fraction:
        comments.append("wide_membrane")

    #flag high average intensity inside the inner membrane border
    if index_border_in > 0 and np.mean(radial_profile_memb[:index_border_in]) >= settings.inside_signal_fraction * peak_height:
        comments.append("high_int_inside")

    #flag high average intensity outside the outer membrane border
    if index_border_out < radial_profile_memb.size and np.mean(radial_profile_memb[index_border_out:]) >= settings.outside_signal_fraction * peak_height:
        comments.append("high_int_outside")

    return peak_radius, peak_index, index_border_in, index_border_out, comments, False

def interpolate_peak_radius(radial_profile, along_radius, peak_index):
    """Refine a discrete peak radius using a quadratic fit over five local samples."""
    if peak_index < 2 or peak_index > radial_profile.size - 3:
        return float(along_radius[peak_index])

    x = along_radius[peak_index - 2:peak_index + 3]
    y = radial_profile[peak_index - 2:peak_index + 3]
    a, b, _ = np.polyfit(x, y, 2)

    #a peak requires a downward-opening parabola
    if a >= 0:
        return float(along_radius[peak_index])

    peak_radius = -b / (2 * a)

    #reject fits whose maximum falls outside the local fitting region
    if peak_radius < x[0] or peak_radius > x[-1]:
        return float(along_radius[peak_index])

    return float(peak_radius)

def detect_noncircular_GUV(intensity_profiles, along_radius, theta, ves_coordinates, settings, channel=0):
    """
    Detect the GUV membrane as a globally smooth radial path using dynamic programming.

    The selected membrane position is chosen by balancing:
        1. high membrane-channel intensity
        2. smoothness between neighboring angles
        3. weak preference for the approximate detected radius

    This allows deformed or elongated GUVs because the radius prior is weak.

    Input
    -----
    intensity_profiles : np.ndarray
        Intensity profiles with shape:
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

    settings : NonCircularMembraneSettings
        Settings used for profile smoothing, contour smoothing, path smoothness,
        radius prior, and the allowed radial search range.

    channel : int
        Channel used for membrane detection.

    Returns
    -------
    shape_x : np.ndarray
        X coordinates of the detected membrane shape.

    shape_y : np.ndarray
        Y coordinates of the detected membrane shape.

    peak_positions : np.ndarray
        Selected radial membrane position for each original angle.

    mean_radius : float
        Mean radius of the detected membrane.

    comments : list
        Quality-control comments associated with the detected contour.

    detection_failed : bool
        True when the membrane could not be detected.
    """

    #extract center and radius from DisGUVery
    xc = ves_coordinates[1]
    yc = ves_coordinates[2]
    approx_radius = ves_coordinates[3]

    #store original angles to return the contour at the original angular resolution
    theta_original = theta.copy()
    original_num_angles = len(theta_original)

    #smooth intensity profiles across neighboring angles
    intensity_profiles_smooth = circular_rolling_average_linear_profiles(intensity_profiles, window_size=settings.profile_smoothing_window)

    #extract membrane channel
    profile_image = intensity_profiles_smooth[:, :, channel]

    #calculate the number of angular profiles used for non-circular detection 
    # --> go to a spacing of 3 px. That helps identify larger jumps when GUVs are together. 
    target_arc_spacing = 3.0
    detection_num_angles = int(np.ceil(2 * np.pi * approx_radius / target_arc_spacing))
    detection_num_angles = min(original_num_angles, np.clip(detection_num_angles, 80, 120))

    #select evenly spaced angular profiles if more profiles are available than needed
    if original_num_angles > detection_num_angles:
        selected_angles = np.linspace(0, original_num_angles, detection_num_angles, endpoint=False).astype(int)
        profile_image = profile_image[:, selected_angles]
        theta = theta[selected_angles]

    #extract number of pixels in the profiles and the number of angles used
    num_radial, num_angles = profile_image.shape

    #define in what region of the profile we are searching for membrane position.
    min_radius = settings.min_radius_fraction * approx_radius
    max_radius = settings.max_radius_fraction * approx_radius
    valid_radius = (along_radius >= min_radius) & (along_radius <= max_radius)

    radial_indices = np.where(valid_radius)[0]
    radius_values = along_radius[radial_indices]

    #if there are not enough pixels, discard GUV.
    if len(radial_indices) < 3:
        shape_x = np.full(original_num_angles, np.nan)
        shape_y = np.full(original_num_angles, np.nan)
        peak_positions = np.full(original_num_angles, np.nan)
        mean_radius = np.nan
        comments = ["insufficient_radial_range"]
        return shape_x, shape_y, peak_positions, mean_radius, comments, True

    #Extract intensities in the allowed region
    membrane_signal = profile_image[radial_indices, :]

    #Check if enough angular profiles have signal above the estimated noise.
    signal_range = np.nanmax(membrane_signal, axis=0) - np.nanmin(membrane_signal, axis=0)
    radial_difference = np.diff(membrane_signal, axis=0)
    noise_level = np.nanmedian(np.abs(radial_difference - np.nanmedian(radial_difference))) / 0.6745

    if np.mean(signal_range > 3 * max(noise_level, 1e-12)) < 0.5:
        empty = np.full(original_num_angles, np.nan)
        return empty.copy(), empty.copy(), empty, np.nan, ["insufficient_membrane_signal"], True

    # Normalize intensity independently for each angle.
    #1. Extract min and max
    signal_min = np.nanmin(membrane_signal, axis=0, keepdims=True)
    signal_max = np.nanmax(membrane_signal, axis=0, keepdims=True)

    #2. minmax normalization
    signal_norm = (membrane_signal - signal_min) / (signal_max - signal_min + 1e-12)

    # create intensity cost: high signal should have low cost to be selected as membrane pixel.
    # it is done by changing the sign to the normalized intensity --> the more intense, the less
    # the cost to be selected.
    node_cost = -signal_norm
    node_cost[~np.isfinite(node_cost)] = 1e6

    # calculate distance from the expected DisGUVery radius
    radius_deviation = radius_values - approx_radius

    # calculate the largest possible distance within the allowed search range
    max_radius_deviation = max(approx_radius - min_radius, max_radius - approx_radius)

    # normalize and square the distance, giving 0 at the expected radius and 1 at the furthest boundary
    radius_prior = (radius_deviation / max_radius_deviation) ** 2

    # add the weighted radius penalty to the intensity cost
    node_cost = node_cost + settings.radius_prior_weight * radius_prior[:, None]

    # Create a 2D matrix where the value at [i, j] is the cost of changing
    # from radius_values[i] at one angle to radius_values[j] at the next angle.
    # Small radius changes have low cost; large jumps have high cost.
    relative_jump = (radius_values[:, None] - radius_values[None, :]) / approx_radius

    # define the relative jump size that should give a normalized penalty of 1
    reference_jump = settings.max_single_jump_fraction

    # calculate the weighted smoothness cost between neighboring angles
    transition_cost = settings.smoothness_weight * (relative_jump / reference_jump) ** 4

    #calculates number of possible radial positions
    num_candidates = len(radius_values)

    #initiates 2 matrices with dimensions number of possible radial positions X number of angles used:
    #cumulative cost: stores cumulative cost for each allowed pixel for each angle
    cumulative_cost = np.zeros((num_candidates, num_angles))

    #backtrack: stores which previous candidate radius has been chosen.
    backtrack = np.zeros((num_candidates, num_angles), dtype=int)

    #since nothing has been chosen yet, no penalty for the distance to the previous pixel has been added.
    # So, it is the cost of the intensity + cost of distance to expected radius.
    cumulative_cost[:, 0] = node_cost[:, 0]

    #for each angle (starting from the second one)
    for angle_i in range(1, num_angles):
        # adds the cost of the previous angle + cost of jumping from the previous position to the new
        total_cost = cumulative_cost[:, angle_i - 1][:, None] + transition_cost

        # For each possible current radius, store pixel position with the lowest cost from the previous angle.
        best_previous_candidate = np.argmin(total_cost, axis=0)

        # Get the corresponding minimum cost.
        best_previous_cost = total_cost[best_previous_candidate, np.arange(num_candidates)]

        backtrack[:, angle_i] = best_previous_candidate

        #store the best total cost for each candidate radius at the current angle.
        cumulative_cost[:, angle_i] = node_cost[:, angle_i] + best_previous_cost

    #create an array to select one pixel per angle
    selected_candidate = np.zeros(num_angles, dtype=int)

    #Since we calculate cumulative cost, the last angle candidate is the one with the lowest cumulative cost
    selected_candidate[-1] = np.argmin(cumulative_cost[:, -1])

    #select following candidates from the last to the first pixel
    # Recover the full best membrane path by moving backward through angles.
    for angle_i in range(num_angles - 2, -1, -1):
        next_angle = angle_i + 1
        next_candidate = selected_candidate[next_angle]
        selected_candidate[angle_i] = backtrack[next_candidate, next_angle]

    # converts selected indices into real radial positions.
    peak_positions_detection = radius_values[selected_candidate]

    # smooth contour
    peak_positions_detection = smooth_membrane_positions(peak_positions_detection, window_size=settings.contour_smoothing_window)

    comments = []

    #calculate the detected mean radius to normalize contour changes independently of the approximate radius
    mean_radius = np.mean(peak_positions_detection)

    #calculate the radius change between neighboring angles, including the jump from the last angle back to the first
    radius_changes = np.diff(peak_positions_detection, append=peak_positions_detection[0])

    #calculate the largest individual radius jump relative to the detected mean radius
    max_jump_fraction = np.max(np.abs(radius_changes)) / mean_radius

    #if one local jump is too large, flag a possible discontinuity in the detected contour
    if max_jump_fraction > settings.max_single_jump_fraction:
        comments.append("contour_jump")

    #calculate how variable the angle-to-angle radius changes are relative to the detected mean radius
    relative_contour_variation = np.std(radius_changes) / mean_radius

    #if the radius changes are too inconsistent, flag the contour as irregular
    if relative_contour_variation > settings.max_contour_variation_fraction:
        comments.append("irregular_contour")

    #interpolate the detected positions back to the original angular resolution
    peak_positions = np.interp(theta_original, theta, peak_positions_detection, period=2 * np.pi)

    # Convert polar contour to XY coordinates
    shape_x = xc + peak_positions * np.cos(theta_original)
    shape_y = yc + peak_positions * np.sin(theta_original)

    return shape_x, shape_y, peak_positions, mean_radius, comments, False


def smooth_membrane_positions(peak_positions, window_size=7):
    """
    Smooth detected membrane positions along the angular direction.

    The angular coordinate is circular, so the first and last angles are treated
    as neighbors.
    """

    # Create angle indices
    num_angles = len(peak_positions)
    angle_index = np.arange(num_angles)

    window_size = int(window_size)

    if window_size <= 1:
        return peak_positions.copy()

    if window_size > num_angles:
        window_size = num_angles

    if window_size % 2 == 0:
        window_size -= 1
        


    # Find valid peak positions
    valid = ~np.isnan(peak_positions)

    # Return unchanged if too few peaks were found
    if np.sum(valid) < 3:
        return peak_positions

    

    # Keep only valid positions
    valid_index = angle_index[valid]
    valid_values = peak_positions[valid]

    # Extend the data so interpolation wraps around the circle
    extended_index = np.concatenate([valid_index - num_angles, valid_index, valid_index + num_angles])

    extended_values = np.concatenate([valid_values, valid_values, valid_values])

    # Fill missing peak positions
    peak_positions_filled = np.interp(angle_index, extended_index, extended_values)

    # Smooth neighboring peak positions
    peak_positions_smooth = uniform_filter1d(peak_positions_filled, size=window_size, mode="wrap")

    return peak_positions_smooth