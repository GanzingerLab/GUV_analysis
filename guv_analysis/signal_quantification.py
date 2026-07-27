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

def localization(
    intensity_profiles,
    peak_index,
    index_border_in,
    index_border_out,
    size_central_area,
    guv_channel=0,
):
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

    for ch, mean_signal, median_signal, channel_rsd in zip(localization_channels, mean_angular_signal, membrane_signal, rsd):
        if mean_signal <= 0:
            # Angular variation cannot be normalized by a non-positive mean signal.
            comment.append(f"zero_mean_ch{ch}")

        if not np.isnan(channel_rsd) and channel_rsd > 0.8:
            # Membrane intensity varies strongly around the vesicle circumference.
            comment.append(f"high_angular_variation_ch{ch}")

        if median_signal <= 0:
            # Localization cannot be normalized by a non-positive membrane signal.
            comment.append(f"zero_median_ch{ch}")

    return localization_index, localization_channels, comment

def localization_from_detected_shape(
    intensity_profiles,
    peak_positions,
    membrane_norm_in=0.95,
    membrane_norm_out=1.05,
    size_central_area=0.25,
    guv_channel=0,
):
    """
    Quantify membrane localization using one detected membrane position
    per angle.

    Both membrane and central intensities are calculated directly from the
    corrected, trimmed intensity profiles.

    Parameters
    ----------
    intensity_profiles : np.ndarray
        Corrected and trimmed profiles with shape
        ``(radial_positions, angles, channels)``.

    peak_positions : np.ndarray
        Detected membrane radial index for each angle. Missing detections
        may be represented by ``np.nan``.

    membrane_norm_in : float, optional
        Inner membrane boundary relative to the detected membrane position.
        For example, 0.95 selects positions starting at 95% of the local
        membrane radius.

    membrane_norm_out : float, optional
        Outer membrane boundary relative to the detected membrane position.
        For example, 1.05 selects positions up to 105% of the local
        membrane radius.

    size_central_area : float, optional
        Size of the central region as a fraction of the local membrane
        position.

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
    num_radial_positions, num_angles, num_channels = intensity_profiles.shape

    if len(peak_positions) != num_angles:
        raise ValueError(
            "peak_positions must contain one membrane position per angle."
        )

    if not 0 <= guv_channel < num_channels:
        raise ValueError("guv_channel is outside the available channel range.")

    if membrane_norm_in >= membrane_norm_out:
        raise ValueError(
            "membrane_norm_in must be smaller than membrane_norm_out."
        )

    localization_channels = np.delete(
        np.arange(num_channels),
        guv_channel,
    )

    radial_indices = np.arange(num_radial_positions)

    # Store one membrane and centre value per angle and protein channel
    angular_membrane_signal = np.full(
        (num_angles, len(localization_channels)),
        np.nan,
        dtype=float,
    )

    angular_centre_signal = np.full(
        (num_angles, len(localization_channels)),
        np.nan,
        dtype=float,
    )

    for angle_i in range(num_angles):
        membrane_position = peak_positions[angle_i]

        # Skip angles without a valid membrane detection
        if np.isnan(membrane_position) or membrane_position <= 0:
            continue

        # Select a band around the local membrane position
        membrane_mask = (
            (radial_indices >= membrane_norm_in * membrane_position)
            & (radial_indices <= membrane_norm_out * membrane_position)
        )

        # Select the central region relative to the local membrane position
        centre_mask = (
            radial_indices <= size_central_area * membrane_position
        )

        if np.any(membrane_mask):
            membrane_values = intensity_profiles[
                membrane_mask,
                angle_i,
                :,
            ][:, localization_channels]

            angular_membrane_signal[angle_i, :] = np.mean(
                membrane_values,
                axis=0,
            )

        if np.any(centre_mask):
            centre_values = intensity_profiles[
                centre_mask,
                angle_i,
                :,
            ][:, localization_channels]

            angular_centre_signal[angle_i, :] = np.mean(
                centre_values,
                axis=0,
            )

    # Combine measurements from all valid angles
    membrane_signal = np.nanmedian(
        angular_membrane_signal,
        axis=0,
    )

    centre_signal = np.nanmean(
        angular_centre_signal,
        axis=0,
    )

    localization_index = np.divide(
        membrane_signal - centre_signal,
        membrane_signal,
        out=np.zeros_like(membrane_signal, dtype=float),
        where=membrane_signal > 0,
    )

    # Measure variation of membrane intensity around the GUV
    mean_angular_signal = np.nanmean(
        angular_membrane_signal,
        axis=0,
    )

    rsd = np.divide(
        np.nanstd(angular_membrane_signal, axis=0),
        mean_angular_signal,
        out=np.full_like(mean_angular_signal, np.nan, dtype=float),
        where=mean_angular_signal > 0,
    )

    comment = []

    for ch, mean_signal, median_signal, channel_rsd in zip(
        localization_channels,
        mean_angular_signal,
        membrane_signal,
        rsd,
    ):
        if mean_signal <= 0 or np.isnan(mean_signal):
            comment.append(f"zero_mean_ch{ch}")

        if not np.isnan(channel_rsd) and channel_rsd > 0.8:
            comment.append(f"high_angular_variation_ch{ch}")

        if median_signal <= 0 or np.isnan(median_signal):
            comment.append(f"zero_median_ch{ch}")

    return localization_index, localization_channels, comment