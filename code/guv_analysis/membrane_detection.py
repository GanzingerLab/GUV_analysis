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

def detect_circular_GUV(radial_profile_memb, radius):
    """
    Detect the membrane inner and outer borders using width at half maximum.
    """
    comments = []

    peaks, _ = find_peaks(
        radial_profile_memb,
        height=10,
        distance=5,
        prominence=1,
    )

    if peaks.size == 0:
        peak_position = 0
        index_border_in  = 0
        index_border_out = 0 
        comment          = ["no_memb_peak"]
        death_mark       = True
        return peak_position, index_border_in, index_border_out, comment, death_mark

    widths = peak_widths(
        radial_profile_memb,
        peaks,
        rel_height=0.5,
    )

    chosen_peak = len(peaks) - 1

    if len(peaks) > 1:
        candidate_indices = range(
            chosen_peak - 1,
            max(-1, chosen_peak - 3),
            -1,
        )

        for candidate in candidate_indices:
            chosen_position = peaks[chosen_peak]
            candidate_position = peaks[candidate]

            chosen_height = radial_profile_memb[chosen_position]
            candidate_height = radial_profile_memb[candidate_position]

            if chosen_position > radius and chosen_height < candidate_height:
                chosen_peak = candidate

        if chosen_peak > 0:  #if the selected peak is not 0, it means there are a lof of peaks inside the GUV --> confetti
            comments.append("confetti")

    peak_position = peaks[chosen_peak]
    peak_height = radial_profile_memb[peak_position]

    index_border_in = int(np.rint(widths[2][chosen_peak]))
    index_border_out = int(np.rint(widths[3][chosen_peak]))

    if index_border_out - index_border_in > radius / 4:
        comments.append("wide_peak")

    if (
        index_border_in > 0
        and np.mean(radial_profile_memb[:index_border_in]) >= 0.3 * peak_height
    ):
        comments.append("high_int_inside")

    if (
        index_border_out < radial_profile_memb.size
        and np.mean(radial_profile_memb[index_border_out:])
        >= 0.4 * peak_height
    ):
        comments.append("high_int_outisde")

    return peak_position , index_border_in, index_border_out, comments, False


def detect_noncircular_GUV(
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
    Detect the GUV membrane as a globally smooth radial path using dynamic programming.

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
    peak_positions = smooth_membrane_positions(
        peak_positions,
        window_size=final_smoothing_window
    )

    # Convert polar contour to XY coordinates
    shape_x = xc + peak_positions * np.cos(theta)
    shape_y = yc + peak_positions * np.sin(theta)

    peak_found = np.full(num_angles, True)

    return shape_x, shape_y, peak_positions, peak_found


def smooth_membrane_positions(peak_positions, window_size=7):
    """
    Smooth detected membrane positions along the angular direction.

    The angular coordinate is circular, so the first and last angles are treated
    as neighbors.
    """

    # Find valid peak positions
    valid = ~np.isnan(peak_positions)

    # Return unchanged if too few peaks were found
    if np.sum(valid) < 3:
        return peak_positions

    # Create angle indices
    num_angles = len(peak_positions)
    angle_index = np.arange(num_angles)

    # Keep only valid positions
    valid_index = angle_index[valid]
    valid_values = peak_positions[valid]

    # Extend the data so interpolation wraps around the circle
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

    # Fill missing peak positions
    peak_positions_filled = np.interp(
        angle_index,
        extended_index,
        extended_values
    )

    # Smooth neighboring peak positions
    peak_positions_smooth = uniform_filter1d(
        peak_positions_filled,
        size=window_size,
        mode="wrap"
    )

    return peak_positions_smooth