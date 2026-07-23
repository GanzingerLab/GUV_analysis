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


def linear_profiles(channels_data, ves_coordinates, parameters_profiles):
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

    num_angles = int(parameters_profiles[0])
    length_excess = parameters_profiles[1]
    dr = parameters_profiles[2]

    profile_radius_limit = length_excess * radius

    # Create evenly spaced angles around the full circle.
    # The angles are expressed in radians.
    theta = np.linspace(0, 2 * np.pi, num_angles, endpoint=False)

    # Create the sampling positions along each radial profile.
    along_radius = np.arange(0, int(profile_radius_limit), dr)

    profile_radius = len(along_radius)

    # Create an empty array with the expected output shape.
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

        return intensity_profiles, along_radius, theta, death_mark

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

    return intensity_profiles, along_radius, theta, death_mark

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
    return np.mean(intensity_profiles, axis=1)