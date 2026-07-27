# %% Imports
import os
import sys
import csv
from glob import glob
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bioio import BioImage
from scipy.ndimage import uniform_filter1d
import tifffile as tif
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "code"))

from guv_analysis.background import background_correction, dilate_vesicle_mask, local_background, mask_all_vesicles
from guv_analysis.io_tools import get_output_folder, load_membrane_positions_json, open_image, reorder_summary_columns, save_membrane_positions, save_shape_normalized_profiles
from guv_analysis.membrane_detection import detect_noncircular_GUV
from guv_analysis.plotting import plot_detected_guv_shape, plot_shape_normalized_profiles_crops_and_angles
from guv_analysis.profiles import angular_profile_from_detected_shape, circular_rolling_average_linear_profiles, linear_profiles, normalized_radial_profile_from_detected_shape
from guv_analysis.signal_quantification import localization_from_detected_shape

# %% Global parameters and functions
GUV_CH = 0
DETECTION_SUFFIX = "_detected_vesicles.csv"

PATH = r"D:\Data\EVOLF"
IMAGE_FORMAT = '.tif'
num_angles = 120
length_excess = 1.5
dr = 1

SIZE_VIEW = length_excess

INNER_MARGIN = 1.2
OUTER_MARGIN = 2.0

PARAMETERS_PROFILES = np.array((num_angles, length_excess, dr))

SMOOTH_WINDOW = 5

DP_SMOOTHNESS_WEIGHT = 0.5
DP_RADIUS_PRIOR_WEIGHT = 0.001
DP_MIN_RADIUS_FRACTION = 0.5
DP_MAX_RADIUS_FRACTION = 1.5
DP_FINAL_SMOOTHING_WINDOW = 9

MEMBRANE_NORM_IN = 0.95
MEMBRANE_NORM_OUT = 1.05

MIN_VALID_SHAPE_FRACTION = 0.9

PLOT_RESULTS = True

pixels_to_remove = 2
size_central_area = 1 / 4

# %% 1. Detect membrane positions and save JSON

images = glob(os.path.join(PATH, "**", "*"+IMAGE_FORMAT), recursive=True)

detection_results = []

for image_i, image_path in tqdm(enumerate(images), total=len(images)):
    print(image_path)
    output_folder = get_output_folder(image_path, PATH)
    membrane_positions = {}

    img = open_image(image_path)

    csv_path = image_path.replace(IMAGE_FORMAT, DETECTION_SUFFIX)

    if not os.path.exists(csv_path):
        print(f"Skipping image. Detection CSV not found: {csv_path}")
        continue

    locs = pd.read_csv(csv_path, index_col=False).to_numpy()

    num_vesicles = len(locs[:, 0])
    dilated_mask = dilate_vesicle_mask(mask_all_vesicles(img, locs))
    num_channels = img.shape[0]

    image_dim = np.array((img.shape[2], img.shape[1]))  # X, Y

    image_results = []

    for ves_i in range(num_vesicles):

        comment = []

        ves_coordinates = locs[ves_i, :]
        vesicle_id = int(ves_coordinates[0])

        # 1. Calculate raw linear profiles from original detected centre
        intensity_profiles, along_radius, theta, death_mark = linear_profiles(
            img,
            ves_coordinates,
            PARAMETERS_PROFILES
        )

        along_radius_ori = along_radius.copy()

        # 2. Skip vesicles too close to the image border
        if death_mark:

            comment = ["margins"]

            image_results.append({
                "image_path": image_path,
                "vesicle_id": vesicle_id,
                "xc": ves_coordinates[1],
                "yc": ves_coordinates[2],
                "radius": ves_coordinates[3],
                "death_mark": True,
                "death_mark_shape": False,
                "valid_shape_fraction": 0,
                "comment": ";".join(comment),
            })

            print("Skipped GUV: profiles extend outside image margins.")
            continue

        # 3. Smooth neighboring angular profiles
        intensity_profiles_smooth = circular_rolling_average_linear_profiles(
            intensity_profiles,
            window_size=SMOOTH_WINDOW
        )

        # 4. Estimate local background using annulus around vesicle
        background = local_background(
            img,
            ves_coordinates,
            dilated_mask,
            inner_margin=INNER_MARGIN,
            outer_margin=OUTER_MARGIN
        )

        # 5. Background correction
        intensity_profiles_smooth_corrected = background_correction(
            intensity_profiles_smooth,
            background
        )

        # 6. Detect membrane shape from GUV channel using global smooth path
        shape_x, shape_y, peak_positions, peak_found = detect_noncircular_GUV(
            intensity_profiles_smooth=intensity_profiles_smooth_corrected,
            along_radius=along_radius_ori,
            theta=theta,
            ves_coordinates=ves_coordinates,
            channel=GUV_CH,
            smoothness_weight=DP_SMOOTHNESS_WEIGHT,
            radius_prior_weight=DP_RADIUS_PRIOR_WEIGHT,
            min_radius_fraction=DP_MIN_RADIUS_FRACTION,
            max_radius_fraction=DP_MAX_RADIUS_FRACTION,
            final_smoothing_window=DP_FINAL_SMOOTHING_WINDOW
        )

        valid_shape = ~np.isnan(shape_x) & ~np.isnan(shape_y)
        valid_shape_fraction = np.mean(valid_shape)

        if valid_shape_fraction < MIN_VALID_SHAPE_FRACTION:

            comment = comment + ["poor_shape_detection"]

            image_results.append({
                "image_path": image_path,
                "vesicle_id": vesicle_id,
                "xc": ves_coordinates[1],
                "yc": ves_coordinates[2],
                "radius": ves_coordinates[3],
                "death_mark": True,
                "death_mark_shape": True,
                "valid_shape_fraction": valid_shape_fraction,
                "comment": ";".join(comment),
            })

            print("Skipped GUV: poor shape detection.")
            continue

        # Store membrane position as list of [x, y] points
        membrane_positions[str(vesicle_id)] = np.column_stack(
            (shape_x, shape_y)
        ).tolist()

        image_results.append({
            "image_path": image_path,
            "vesicle_id": vesicle_id,
            "xc": ves_coordinates[1],
            "yc": ves_coordinates[2],
            "radius": ves_coordinates[3],
            "death_mark": False,
            "death_mark_shape": False,
            "valid_shape_fraction": valid_shape_fraction,
            "comment": ";".join(comment),
        })

        # Optional visual check of detected shape
        image_stem = Path(image_path).stem 

        shape_plot_path = os.path.join(
            output_folder,
            f"{image_stem}_ves{vesicle_id}_membrane_detection.png"
        )

        plot_detected_guv_shape(
            channels_data=img,
            GUV_channel=GUV_CH,
            ves_coordinates=ves_coordinates,
            shape_x=shape_x,
            shape_y=shape_y,
            size_view=SIZE_VIEW,
            title=f"Detected GUV shape - vesicle {vesicle_id}",
            save_path=shape_plot_path,
            show=PLOT_RESULTS
        )

    membrane_path = save_membrane_positions(
        output_folder=output_folder,
        image_path=image_path,
        membrane_positions=membrane_positions
    )

    for result_row in image_results:
        result_row["membrane_position_file"] = membrane_path
        detection_results.append(result_row)


detection_results_df = pd.DataFrame(detection_results)

detection_results_path = os.path.join(PATH, "membrane_detection_results.csv")
detection_results_df.to_csv(detection_results_path, index=False)

print(f"\nDone. Membrane detection results saved to: {detection_results_path}")


# %% EXAMPLE Load membrane JSON and plot GUV outline

membrane_path = r"D:\Data\EVOLF\output\20260617_GUV_FLuo4_TRIMEB_aHL_perm_2_test_analysis_radial_profile\20260610_D4_t0-04_membrane_positions.json"

source_file, vesicles = load_membrane_positions_json(membrane_path)

print(source_file)
print(vesicles.keys())

vesicle_id = "1"

membrane_xy = vesicles[vesicle_id]

membrane_xy_closed = np.vstack([
    membrane_xy,
    membrane_xy[0, :]
])

plt.figure()
plt.plot(membrane_xy_closed[:, 0], membrane_xy_closed[:, 1], ".-")
plt.axis("equal")
plt.xlabel("x")
plt.ylabel("y")
plt.title(f"GUV outline - vesicle {vesicle_id}")
plt.show()


# %% 2. Shape-normalized radial profile and plotting for all saved membrane JSON files

# Find all saved membrane-position JSON files
membrane_jsons = glob(
    os.path.join(PATH, "output", "**", "*membrane_positions.json"),
    recursive=True
)

# Find all original CZI files once
images = glob(os.path.join(PATH, "**", "*"+IMAGE_FORMAT), recursive=True)

# Store results from all images and all GUVs in one list
profile_results = []

for membrane_path in tqdm(membrane_jsons):

    print(f"\nProcessing membrane file: {membrane_path}")

    source_file, vesicles = load_membrane_positions_json(membrane_path)
    image_profile_results = []
    profiles_data = {}

    image_name = source_file
    image_path = None

    for candidate_image_path in images:
        if os.path.basename(candidate_image_path) == image_name:
            image_path = candidate_image_path
            break

    if image_path is None:
        print(f"Skipping. Could not find source image: {image_name}")
        continue

    print(image_path)

    output_folder = get_output_folder(image_path, PATH)

    img = open_image(image_path)

    csv_path = image_path.replace(IMAGE_FORMAT, DETECTION_SUFFIX)

    if not os.path.exists(csv_path):
        print(f"Skipping image. Detection CSV not found: {csv_path}")
        continue

    locs = pd.read_csv(csv_path, index_col=False).to_numpy()
    dilated_mask = dilate_vesicle_mask(mask_all_vesicles(img, locs))

    num_channels = img.shape[0]

    image_dim = np.array((img.shape[2], img.shape[1]))  # X, Y

    for vesicle_id, membrane_xy in vesicles.items():

        vesicle_id_int = int(vesicle_id)

        matching_rows = locs[locs[:, 0].astype(int) == vesicle_id_int]

        if len(matching_rows) == 0:
            print(f"Skipping vesicle {vesicle_id}: not found in detection CSV.")
            continue

        ves_coordinates = matching_rows[0, :]

        # 1. Calculate raw linear profiles from original detected centre
        intensity_profiles, along_radius, theta, death_mark = linear_profiles(
            img,
            ves_coordinates,
            PARAMETERS_PROFILES
        )

        along_radius_ori = along_radius.copy()

        if death_mark:
            print(f"Skipping vesicle {vesicle_id}: profiles extend outside image margins.")
            continue

        # 2. Smooth neighboring angular profiles
        intensity_profiles_smooth = circular_rolling_average_linear_profiles(
            intensity_profiles,
            window_size=SMOOTH_WINDOW
        )

        # 3. Estimate local background using annulus around vesicle
        background = local_background(
            img,
            ves_coordinates,
            dilated_mask,
            inner_margin=INNER_MARGIN,
            outer_margin=OUTER_MARGIN
        )

        # 4. Background correction
        intensity_profiles_smooth_corrected = background_correction(
            intensity_profiles_smooth,
            background
        )

        # Convert saved XY membrane contour back to radial membrane distance
        xc = ves_coordinates[1]
        yc = ves_coordinates[2]

        peak_positions = np.sqrt(
            (membrane_xy[:, 0] - xc) ** 2
            + (membrane_xy[:, 1] - yc) ** 2
        )

        # 5. Create normalized radial profile using local membrane distance
        radial_profiles_normalized, normalized_distance_axis, normalized_profiles = (
            normalized_radial_profile_from_detected_shape(
                intensity_profiles=intensity_profiles_smooth_corrected,
                along_radius=along_radius_ori,
                peak_positions=peak_positions,
            )
        )

        # 6. Shape-normalized localization
        localization_index, localization_channels, comment_loc = (
            localization_from_detected_shape(
                intensity_profiles=intensity_profiles_smooth_corrected,
                peak_positions=peak_positions,
                membrane_norm_in=MEMBRANE_NORM_IN,
                membrane_norm_out=MEMBRANE_NORM_OUT,
                size_central_area=size_central_area,
                guv_channel=GUV_CH
            )
        )

        angular_profiles_normalized = angular_profile_from_detected_shape(intensity_profiles_smooth_corrected, along_radius_ori, peak_positions)

        profiles_data[f"ves{vesicle_id}_radial_profiles_normalized"] = radial_profiles_normalized
        profiles_data[f"ves{vesicle_id}_normalized_distance_axis"] = normalized_distance_axis
        profiles_data[f"ves{vesicle_id}_angular_profiles_normalized"] = angular_profiles_normalized

        image_stem = Path(image_path).stem

        combined_plot_path = os.path.join(
            output_folder,
            f"{image_stem}_ves{vesicle_id}_shape_normalized_profiles.png"
        )

        plot_shape_normalized_profiles_crops_and_angles(
            channels_data=img,
            ves_coordinates=ves_coordinates,
            radial_profiles_normalized=radial_profiles_normalized,
            normalized_distance_axis=normalized_distance_axis,
            angular_profiles_normalized=angular_profiles_normalized,
            channels=tuple(range(num_channels)),
            size_view=SIZE_VIEW,
            title=f"Shape-normalized profiles - vesicle {vesicle_id}",
            save_path=combined_plot_path,
            show=PLOT_RESULTS
        )

        result_row = {
            "image_path": image_path,
            "membrane_position_file": membrane_path,
            "vesicle_id": vesicle_id_int,
            "xc": ves_coordinates[1],
            "yc": ves_coordinates[2],
            "radius": ves_coordinates[3],
            "comment": ";".join(comment_loc),
        }

        for ch in range(num_channels):
            result_row[f"background_ch{ch}"] = background[ch]

        for loc_value, ch in zip(localization_index, localization_channels):
            result_row[f"localization_ch{ch}"] = loc_value

        image_profile_results.append(result_row)
    
    profile_data_path = save_shape_normalized_profiles(
        output_folder=output_folder,
        image_path=image_path,
        profiles_data=profiles_data
    )

    for result_row in image_profile_results:
        result_row["profile_data_file"] = profile_data_path
        profile_results.append(result_row)


# Save one CSV with all GUVs from all images
profile_results_df = pd.DataFrame(profile_results)
profile_results_df = reorder_summary_columns(profile_results_df)

profile_results_path = os.path.join(PATH, "shape_normalized_profile_results.csv")
profile_results_df.to_csv(profile_results_path, index=False)

print(f"\nDone. Shape-normalized profile results saved to: {profile_results_path}")