# %%
import os
import sys
import csv
from glob import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bioio import BioImage
from scipy.signal import find_peaks, peak_widths
import tifffile as tif
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "code"))

from guv_analysis.background import background_correction, dilate_vesicle_mask, local_background, mask_all_vesicles
from guv_analysis.io_tools import get_output_folder, open_image
from guv_analysis.membrane_detection import detect_circular_GUV
from guv_analysis.plotting import plot_profile_and_zoom, save_separate_profile_plots
from guv_analysis.profiles import angular_profile, linear_profiles, radial_profile, trim_central_profiles
from guv_analysis.signal_quantification import localization
#%%
# Parameters

GUV_CH = 0
DETECTION_SUFFIX = "_detected_vesicles.csv"

PATH = r"D:\Data\EVOLF"
IMAGE_FORMAT = '.czi'
num_angles = 360
length_excess = 1.5
dr = 1

SIZE_VIEW = length_excess

INNER_MARGIN = 1.2
OUTER_MARGIN = 2.0

PARAMETERS_PROFILES = np.array((num_angles, length_excess, dr))

PLOT_RESULTS = True

pixels_to_remove = 2
size_central_area = 1/4  # the central area radius (unit of measure: each vesicles' radius) considered during localization quantification
#%%
# Run analysis
images = glob(os.path.join(PATH, "**", "*"+IMAGE_FORMAT), recursive=True)
all_results = []
for image_i, image_path in tqdm(enumerate(images), total=len(images),):
    print(image_path)
    comment = []
    # Open data for each image
    # get unique part of the path in case we want to save the results in an output folder. 

    output_folder = get_output_folder(image_path, PATH)

    img = open_image(image_path) #open image

    #changes .czi to the suffix you had set and if the CSv exists it opens it
    csv_path = image_path.replace(IMAGE_FORMAT, DETECTION_SUFFIX)

    if not os.path.exists(csv_path):
        print(f"Skipping image. Detection CSV not found: {csv_path}")
        continue

    locs = pd.read_csv(csv_path, index_col=False).to_numpy()

    num_vesicles = len(locs[:, 0])
    dilated_mask = dilate_vesicle_mask(mask_all_vesicles(img, locs))
    num_channels = img.shape[0]

    image_dim = np.array((img.shape[2], img.shape[1]))  # X, Y
   
    # Vesicle-level analysis
    for ves_i in range(num_vesicles):
        ves_coordinates = locs[ves_i, :]
        vesicle_id = ves_coordinates[0]

        # 1. Calculate linear profiles
        intensity_profiles, along_radius, theta, death_mark = linear_profiles(img, ves_coordinates, PARAMETERS_PROFILES)
        intensity_profiles, along_radius = trim_central_profiles(intensity_profiles, along_radius, pixels_to_remove)
        # 2. Skip vesicles too close to the image border
        if death_mark:
            comment = ['margins']

            result_row = {
                "image_path": image_path,
                "vesicle_id": vesicle_id,
                "xc": ves_coordinates[1],
                "yc": ves_coordinates[2],
                "radius": ves_coordinates[3],
                "death_mark": death_mark,
                "comment": ";".join(comment),
            }

            all_results.append(result_row)

            print("Skipped GUV: profiles extend outside image margins.")
            continue

        # 3. Estimate local background using annulus around vesicle
        background = local_background(
            img,
            ves_coordinates,
            dilated_mask,
            inner_margin=INNER_MARGIN,
            outer_margin=OUTER_MARGIN
        )

        # 4. Background correction
        intensity_profiles_corrected = background_correction(
            intensity_profiles,
            background
        )

        # 5. Calculate and correct radial profiles
        radial_profiles = radial_profile(
            intensity_profiles_corrected
        )

        peak_index, index_border_in, index_border_out, comment_peak, death_mark_peak = detect_circular_GUV(radial_profiles[:,GUV_CH], locs[ves_i,3])

        ## Registration of the triggered conditions for automatic vesicle rejection:
        if comment_peak:
            comment = comment + comment_peak
        ## Output for profiles where no peak was detected is set to 0:
        if death_mark_peak:
            background = np.zeros((num_channels))
            localization_values = np.zeros(num_channels - 1)
            comment = comment_peak

            result_row = {
                "image_path": image_path,
                "vesicle_id": vesicle_id,
                "xc": ves_coordinates[1],
                "yc": ves_coordinates[2],
                "radius": ves_coordinates[3],
                "death_mark": True,
                "comment": ";".join(comment),
            }


            all_results.append(result_row)

            print("Skipped GUV: no membrane peak detected.")
            continue

        # 6. Calculation of the angular profile of the vesicle under study:
        angular_profiles = angular_profile(intensity_profiles_corrected, index_border_in, index_border_out)
        
        # 7. Quantification of protein localization on the membrane:
        localization_index, localization_channels, comment_loc = localization(
                intensity_profiles_corrected,
                peak_index,
                index_border_in,
                index_border_out,
                size_central_area,
                guv_channel=GUV_CH
            )
        if comment_loc:
            comment = comment + comment_loc
        # 8. Plot result for visual inspection
        if PLOT_RESULTS:
            channels_to_plot = tuple(range(num_channels))

            full_plot_path = os.path.join(
                output_folder,
                f"full_ves{vesicle_id}.png"
            )

            plot_profile_and_zoom(
                channels_data=img,
                ves_coordinates=ves_coordinates,
                radial_profiles=radial_profiles,
                along_radius=along_radius,
                angular_profiles=angular_profiles,
                size_view=SIZE_VIEW,
                channels=channels_to_plot,
                save_path=full_plot_path,
                show=PLOT_RESULTS
            )

            save_separate_profile_plots(
                radial_profiles=radial_profiles,
                along_radius=along_radius,
                angular_profiles=angular_profiles,
                vesicle_id=vesicle_id,
                channels=channels_to_plot,
                output_folder=output_folder
            )

        # 8. Store summary result
        result_row = {
            "image_path": image_path,
            "vesicle_id": vesicle_id,
            "xc": ves_coordinates[1],
            "yc": ves_coordinates[2],
            "radius": ves_coordinates[3],
            "death_mark": death_mark,
            "death_mark_peak": death_mark_peak,
            "index_border_in": index_border_in,
            "index_border_out": index_border_out,
            "comment": ";".join(comment),
        }

        for ch in range(num_channels):
            result_row[f"background_ch{ch}"] = background[ch]

        for loc_value, ch in zip(localization_index, localization_channels):
            result_row[f"localization_ch{ch}"] = loc_value

        all_results.append(result_row)


print(f"\nDone. Results saved to:")

results_df = pd.DataFrame(all_results)
results_df.to_csv(os.path.join(PATH, 'results_2.csv'), index=False)