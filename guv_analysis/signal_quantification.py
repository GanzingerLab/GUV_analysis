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

def circular_membrane_localization(intensity_profiles, peak_index, index_border_in, index_border_out, size_central_area, guv_channel=0):
    """
    Quantify membrane localization for all non-GUV channels.

    Both membrane and central intensities are calculated directly from the
    corrected, trimmed intensity profiles.

    Parameters
    ----------
    intensity_profiles : np.ndarray
        Corrected and trimmed profiles with shape
        ``(radial_positions, angles, channels)``.

    peak_index : int
        Radial index of the detected membrane peak.

    index_border_in : int
        Inner membrane-border index.

    index_border_out : int
        Outer membrane-border index.

    size_central_area : float
        Size of the central region as a fraction of the distance from the first
        retained radial position to the membrane peak.

    guv_channel : int, optional
        Membrane-marker channel, excluded from localization quantification.

    Returns
    -------
    localization_index : np.ndarray
        Localization values for all non-GUV channels.

    localization_channels : np.ndarray
        Channel indices corresponding to ``localization_index``.

    comment : list of str
        Quality-control comments.
    """
    localization_channels = np.delete(np.arange(intensity_profiles.shape[2]), guv_channel)

    index_centre = int(size_central_area * peak_index)

    membrane_values = intensity_profiles[index_border_in:index_border_out, :, localization_channels]

    centre_values = intensity_profiles[:index_centre, :, localization_channels]

    membrane_signal = np.median(membrane_values, axis=(0, 1))

    centre_signal = np.mean(centre_values, axis=(0, 1))

    localization_index = np.divide(
        membrane_signal - centre_signal,
        membrane_signal,
        out=np.zeros_like(membrane_signal, dtype=float),
        where=membrane_signal > 0,
    )

    comment = []

    angular_membrane_signal = np.mean(membrane_values, axis=0)

    mean_angular_signal = np.mean(angular_membrane_signal, axis=0)

    rsd = np.divide(
        np.std(angular_membrane_signal, axis=0),
        mean_angular_signal,
        out=np.full_like(mean_angular_signal, np.nan, dtype=float),
        where=mean_angular_signal > 0,
    )

    localization = {int(ch): float(value) for ch, value in zip(localization_channels, localization_index)}
    return localization, comment

def noncircular_membrane_localization(intensity_profiles, along_radius, peak_radius_by_angle, membrane_width_samples=5, size_central_area=0.25, guv_channel=0):
    """
    Quantify membrane localization using one detected membrane radius per angle.

    Both membrane and central intensities are calculated directly from the
    corrected, trimmed intensity profiles.

    Parameters
    ----------
    intensity_profiles : np.ndarray
        Corrected and trimmed profiles with shape
        ``(radial_positions, angles, channels)``.

    along_radius : np.ndarray
        Radial distance corresponding to the first axis of
        ``intensity_profiles``.

    peak_radius_by_angle : np.ndarray
        Detected membrane radius for each angle. Missing detections may be
        represented by ``np.nan``.

    membrane_width_samples : int, optional
        Odd number of radial samples averaged around each detected membrane
        position. For example, 5 uses the closest membrane sample and 2 samples
        on each side.

    size_central_area : float, optional
        Size of the central region as a fraction of the local membrane radius.

    guv_channel : int, optional
        Membrane-marker channel, excluded from localization quantification.

    Returns
    -------
    localization_index : np.ndarray
        Localization values for all non-GUV channels.

    localization_channels : np.ndarray
        Channel indices corresponding to ``localization_index``.

    comments : list[str]
        Quality-control comments.
    """
    num_radial_positions, num_angles, num_channels = intensity_profiles.shape

    if along_radius.size != num_radial_positions:
        raise ValueError("along_radius must match the radial dimension of intensity_profiles.")

    if peak_radius_by_angle.size != num_angles:
        raise ValueError("peak_radius_by_angle must contain one membrane radius per angle.")

    if membrane_width_samples < 1 or membrane_width_samples % 2 == 0:
        raise ValueError("membrane_width_samples must be a positive odd number.")

    if not 0 <= guv_channel < num_channels:
        raise ValueError("guv_channel is outside the available channel range.")

    localization_channels = np.delete(np.arange(num_channels), guv_channel)
    angular_membrane_signal = np.full((num_angles, len(localization_channels)), np.nan, dtype=float)
    angular_centre_signal = np.full((num_angles, len(localization_channels)), np.nan, dtype=float)
    half_width = membrane_width_samples // 2

    for angle_i, membrane_radius in enumerate(peak_radius_by_angle):
        # Skip angles without a valid membrane detection
        if np.isnan(membrane_radius) or membrane_radius <= 0:
            continue

        # Find the radial sample closest to the detected membrane radius
        peak_index = np.argmin(np.abs(along_radius - membrane_radius))

        # Select the same number of radial samples inside and outside the membrane
        start_index = peak_index - half_width
        end_index = peak_index + half_width + 1

        # Skip angles where the complete requested membrane window does not fit
        if start_index < 0 or end_index > num_radial_positions:
            continue

        # Select the central region relative to the local membrane radius
        centre_mask = along_radius <= size_central_area * membrane_radius

        membrane_values = intensity_profiles[start_index:end_index, angle_i, :][:, localization_channels]
        angular_membrane_signal[angle_i, :] = np.mean(membrane_values, axis=0)

        if np.any(centre_mask):
            centre_values = intensity_profiles[centre_mask, angle_i, :][:, localization_channels]
            angular_centre_signal[angle_i, :] = np.mean(centre_values, axis=0)

    # Combine measurements from all valid angles
    membrane_signal = np.nanmedian(angular_membrane_signal, axis=0)
    centre_signal = np.nanmean(angular_centre_signal, axis=0)

    localization_index = np.divide(
        membrane_signal - centre_signal,
        membrane_signal,
        out=np.zeros_like(membrane_signal, dtype=float),
        where=membrane_signal > 0,
    )

    # Measure variation of membrane intensity around the GUV
    mean_angular_signal = np.nanmean(angular_membrane_signal, axis=0)
    rsd = np.divide(
        np.nanstd(angular_membrane_signal, axis=0),
        mean_angular_signal,
        out=np.full_like(mean_angular_signal, np.nan, dtype=float),
        where=mean_angular_signal > 0,
    )

    comments = []

    localization = {int(ch): float(value) for ch, value in zip(localization_channels, localization_index)}
    return localization, comments

def calculate_inside_intensity(img, mask, background):
    """
    Calculate raw and background-corrected mean intensities inside a GUV mask.

    Parameters
    ----------
    img : np.ndarray
        Image with shape ``(channels, height, width)``.

    mask : np.ndarray
        Two-dimensional boolean mask defining the GUV interior.

    background : np.ndarray
        Background intensity for each image channel.

    Returns
    -------
    raw_intensity : np.ndarray
        Mean raw intensity inside the mask for every channel.

    corrected_intensity : np.ndarray
        Mean intensity inside the mask after background subtraction.
    """
    if mask.shape != img.shape[1:]:
        raise ValueError("mask must match the image height and width.")

    if not np.any(mask):
        raise ValueError("GUV mask contains no pixels.")

    raw_intensity = np.mean(img[:, mask], axis=1)
    corrected_intensity = raw_intensity - background
    return raw_intensity, corrected_intensity