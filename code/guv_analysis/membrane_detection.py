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

def membrane_detection(radial_profile_memb, radius):
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
        index_border_in  = 0
        index_border_out = 0 
        comment          = ["no_memb_peak"]
        death_mark       = True
        return index_border_in, index_border_out, comment, death_mark

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
