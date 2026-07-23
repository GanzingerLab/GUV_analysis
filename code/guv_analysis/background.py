
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
from scipy.ndimage import binary_dilation

def mask_all_vesicles(img, detection):
    """
    Create a binary mask containing all detected vesicles.
    
    INPUTS:
    img: 3D numpy array of the image (channels, height, width)
    detection: 2D numpy array of detection results (vesicle_id, x, y, r)

    RETURNS:
    mask_all_vesicles: 2D boolean numpy array of the same height and width as the image, True = presence of a vesicle.
    """
    height, width = img.shape[1:]
    mask = np.zeros((height, width), dtype=bool)

    circles = np.rint(detection[:, 1:4]).astype(int)

    for xc, yc, radius in circles:
        if radius < 0:
            continue

        # Clip the circle's bounding box to the image.
        x0 = max(0, xc - radius)
        x1 = min(width, xc + radius + 1)
        y0 = max(0, yc - radius)
        y1 = min(height, yc + radius + 1)

        if x0 >= x1 or y0 >= y1:
            continue

        yy, xx = np.ogrid[y0:y1, x0:x1]
        circle = (xx - xc) ** 2 + (yy - yc) ** 2 <= radius ** 2

        mask[y0:y1, x0:x1] |= circle

    return mask

def background_correction(num_channels, intensity_profiles, background):
    """ 
    Background correction of the intesity linear profiles. 
    
    Input
    -----
    num_channels : int
        The number of channels present in the image analyzed.
        
    intensity_profiles : np.ndarray
        The intensity linear profiles calculated.
        Note: Data related to different channels are stored along the 3rd 
              dimension of the array. Element [:,:,0] is always based on the 
              membrane channel, element [:,:,1] is based on the single protein
              present. If both proteins are present, element [:,:,1] corresponds
              to the septin channel and element [:,:,2] corresponds to the actin
              channel.
              
    background : np.ndarray
        The background noise estimated for the different channels
        Data related to different 
        channels are stored along the 3rd dimension of the array:
        background[:,0] always corresponds to the background of the membrane channel.
        If only one protein is present:
        background[:,1] corresponds to the background of that protein's channel.
        If two proteins are present:
        background[:,1] corresponds to the background of the septin channel.    
        background[:,2] corresponds to the background of the actin channel.
    
    Returns
    -------
    intensity_profiles_corrected : np.ndarray
        The intensity linear profiles after the background noise correction.
        Note: Data is stored as in the intensity_profiles array. 
              
    """
    intensity_profiles_corrected = np.zeros_like(intensity_profiles)

    for i in range(num_channels):
        intensity_profiles_corrected[:, :, i] = np.clip(
            intensity_profiles[:, :, i] - background[0, i],
            0,
            None
        )

    return intensity_profiles_corrected

def global_background(img, mask):
    """
    Estimate global background all pixels outside GUVs (mask indicates GUVs)

    Uses pixels between inner_margin * radius and outer_margin * radius.
    """
    structure = np.ones((3, 3), dtype=bool)
    dilated_mask = binary_dilation(mask, structure=structure, iterations=3)
    return np.mean(img[:,~dilated_mask], axis=1)

def local_background(
    img,
    ves_coordinates,
    global_mask,
    inner_margin=1.2,
    outer_margin=2.0,  
):
    """
    Estimate local background from an annulus around the vesicle.

    Uses pixels between inner_margin * radius and outer_margin * radius.
    """
    xc = ves_coordinates[1]
    yc = ves_coordinates[2]
    radius = ves_coordinates[3]

    y, x = np.indices(img.shape[1:])

    distance = np.sqrt((x - xc) ** 2 + (y - yc) ** 2)

    structure = np.ones((3, 3), dtype=bool)
    dilated_mask = binary_dilation(global_mask, structure=structure, iterations=3)

    background_mask = (
        (distance >= inner_margin * radius)
        & (distance <= outer_margin * radius)
        & (~dilated_mask)
    )
    if not np.any(background_mask):
        raise ValueError(f"No background pixels found for vesicle {ves_coordinates[0]}")
    elif np.count_nonzero(background_mask) < 10:
        raise ValueError(f"Too few background pixels for vesicle {ves_coordinates[0]}")
        

    background = np.mean(img[:, background_mask], axis=1)

    return background