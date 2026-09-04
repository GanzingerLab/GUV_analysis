import numpy as np
from skimage.measure import label
from skimage.color import label2rgb
from scipy import ndimage
from scipy.signal import find_peaks, peak_widths

def filter_clumped_vesicles(detection, mask):
    """
    Filter out vesicles that are part of a lipid clump.

    INPUTS:
    detection: 2D numpy array of detection results (vesicle_id, x, y, r)
    mask: 2D boolean numpy array of the same height and width as the image, True = presence of a vesicle.
    df: pandas DataFrame with detection results and a column for filter comments.

    RETURNS:
    df : pandas DataFrame with filtered detection results and a column indicating if a vesicle was part of a lipid clump.
    """
    detection_int = np.rint(detection).astype(int)
    
    labels = label(mask)
    size_labels = ndimage.sum(mask, labels, range(1, labels.max() + 1))
    r_max = np.max(detection_int[:, 3])
    max_vesicle_area = np.pi * r_max**2
    lipid_clump = size_labels > 2*max_vesicle_area
    clump_mask = np.isin(labels, np.where(lipid_clump)[0] + 1)  # +1 because labels are 1-indexed
    vesicles_clumped = clump_mask[detection[:,2].astype(int), detection[:,1].astype(int)]
    return detection[:, 0][vesicles_clumped]

def membrane_fraction(intensity_profiles, peak_index_by_angle, global_background, profile_settings, guv_channel=0, baseline_gap=3):
    """
    Evaluate whether a detected membrane contour is supported by localized
    radial intensity peaks.

    At each angle, the function compares the strongest intensity near the
    detected membrane index with nearby radial regions inside and outside the
    membrane.

    The membrane prominence is normalized by a robust estimate of the local
    intensity variation:

        support = membrane_prominence / local_noise

    The function then calculates ``fraction_membrane``: the fraction of valid
    angles whose support is greater than or equal to
    ``profile_settings.min_radial_membrane_support``.

    The candidate is death-marked when ``fraction_membrane`` is below
    ``profile_settings.min_fraction_membrane``.

    Parameters
    ----------
    intensity_profiles : np.ndarray
        Radial intensity profiles with shape
        ``(radial_positions, angles, channels)``.

    peak_index_by_angle : float or array-like
        Detected radial membrane index.

        For circular analysis, this may be a scalar. The same membrane index is
        then used for every angle.

        For noncircular analysis, this must contain one radial index per angle.
        Invalid detections may be represented by ``np.nan``.

    global_background : float
        Global background intensity. It is used as the minimum denominator when
        the local MAD-based intensity variation is very small.

    profile_settings : ProfileSettings
        Profile-analysis settings. The function uses:

        - ``noncircular_membrane_width_pixels``
        - ``min_radial_membrane_support``
        - ``min_fraction_membrane``

    guv_channel : int, optional
        Index of the channel containing the membrane marker.

    baseline_gap : int, optional
        Number of radial samples skipped between the membrane window and each
        baseline window. The default is 3.

    Returns
    -------
    support : np.ndarray
        Radial membrane-support score for every angle. Higher values indicate
        stronger evidence for a localized membrane peak.

    prominence : np.ndarray
        Difference between the strongest membrane-window intensity and the
        brighter of the inner and outer baseline intensities.

    fraction_membrane : float
        Fraction of valid angles whose support is greater than or equal to
        ``profile_settings.min_radial_membrane_support``.

    death_mark : bool
        ``True`` when ``fraction_membrane`` is below
        ``profile_settings.min_fraction_membrane``. It is also ``True`` when no
        valid support values can be calculated.

    Notes
    -----
    The membrane and baseline windows use the same number of radial samples,
    defined by ``profile_settings.noncircular_membrane_width_pixels``.

    Support remains np.nan for angles where the membrane index is invalid, 
    the membrane or inner baseline window does not fit, no outer baseline 
    pixels are available, or one of the windows contains no finite values.
    """
    channel_profiles = intensity_profiles[:, :, guv_channel]
    num_radial_positions, num_angles = channel_profiles.shape

    membrane_width = (profile_settings.noncircular_membrane_width_pixels)
    if membrane_width < 1 or membrane_width % 2 == 0:
        raise ValueError("`noncircular_membrane_width_pixels` must be a positive odd number.")

    if baseline_gap < 0:
        raise ValueError("`baseline_gap` must be non-negative.")

    membrane_half_width = membrane_width // 2
    baseline_width = membrane_width

    peak_index_by_angle = np.asarray(peak_index_by_angle, dtype=float)

    if peak_index_by_angle.ndim == 0:
        peak_index_by_angle = np.full(
            num_angles,
            peak_index_by_angle.item(),
            dtype=float,
        )

    support = np.full(num_angles, np.nan, dtype=float)
    prominence = np.full(num_angles, np.nan, dtype=float)

    noise_floor = max(float(global_background), 1e-6)

    for angle_i, detected_peak_index in enumerate(peak_index_by_angle):
        # Skip angles without a valid detected membrane index.
        if not np.isfinite(detected_peak_index):
            continue

        peak_index = int(round(detected_peak_index))

        # Membrane window centred on the detected radial index.
        membrane_start = peak_index - membrane_half_width
        membrane_end = peak_index + membrane_half_width + 1

        # Baseline region inside the membrane.
        inner_end = membrane_start - baseline_gap
        inner_start = inner_end - baseline_width
        inner_start = max(inner_start, 0)

        # Baseline region outside the membrane.
        outer_start = membrane_end + baseline_gap
        outer_end = outer_start + baseline_width

        radial_profile = channel_profiles[:, angle_i]

        outer_end = min(outer_end, num_radial_positions)

        # All three windows must fit inside the radial profile.
        if inner_start < 0 or outer_start >= outer_end:
            continue

        membrane_values = radial_profile[membrane_start:membrane_end]
        inner_values = radial_profile[inner_start:inner_end]
        outer_values = radial_profile[outer_start:outer_end]

        # The inner baseline must fit completely, and at least one outer-baseline pixel must be available.
        if (
            not np.any(np.isfinite(membrane_values))
            or not np.any(np.isfinite(inner_values))
            or not np.any(np.isfinite(outer_values))):
            continue

        # Strongest intensity near the detected membrane position.
        membrane_signal = np.nanmax(membrane_values)

        # Typical intensity immediately inside and outside the membrane.
        inner_signal = np.nanmedian(inner_values)
        outer_signal = np.nanmedian(outer_values)

        # Compare the membrane with the brighter surrounding side.
        local_baseline = max(inner_signal, outer_signal)

        membrane_prominence = (membrane_signal - local_baseline)
        prominence[angle_i] = membrane_prominence

        # Combine both baseline regions to estimate local variation.
        baseline_values = np.concatenate((inner_values, outer_values))

        baseline_median = np.nanmedian(baseline_values)

        # Median absolute deviation, robust to isolated bright or dark values.
        baseline_mad = np.nanmedian(np.abs(baseline_values - baseline_median))

        # Convert MAD to a standard-deviation-like scale and prevent a very
        # small denominator by using the global background as a noise floor.
        #ref: https://www.itl.nist.gov/div898/software/dataplot/refman2/auxillar/mad.htm?utm_source=chatgpt.com
        local_noise = max(1.4826 * baseline_mad, noise_floor)

        support[angle_i] = (membrane_prominence / local_noise)

    valid = np.isfinite(support)

    if np.any(valid):
        supported = (support[valid] >= profile_settings.min_radial_membrane_support)
        fraction_membrane = float(np.mean(supported))
    else:
        fraction_membrane = 0.0

    death_mark = (fraction_membrane < profile_settings.min_fraction_membrane)

    return support, prominence, fraction_membrane, death_mark

