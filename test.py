#%%
from guv_analysis.settings import AnalysisSettings
from guv_analysis.image import GUVImage
from guv_analysis.tif_converter import czi_to_tif
from guv_analysis.hdf_manager import load_hdf5
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
#%%Transform images from czi to tif, preserving part of the metadata
path = Path(r"D:\Data\EVOLF\test")

czi_files = list(path.rglob("*.czi"))

czi = czi_files[0]

for czi in czi_files:
    print("Converting:", czi)
    output = czi_to_tif(czi)
    print("Saved:", output)
#%%
# User settings

image_path = r"P:\26 Optimizing PURE encapsulation and function in GUV Franzi\analysis\disguvery\testing\20260702_Exp_36_wellscans_N5_19H_1024px_#01_cropped_tiffs\test\tile_r01_c01.tif"
analysis_route = "noncircular"  # Options: "circular", "noncircular"
guv_id_to_test = 1
save_results = True

settings = AnalysisSettings()
#######IMPORTANT TO CHECK THE guv_ch########
settings.guv_ch = 1 
settings.background.method = "local_then_global"  # Options: "local", "global", "local_then_global"
settings.profiles.length_excess = 1.5
settings.plotting.show = True


#%%
# 1. Load image and detections

image_analysis = GUVImage(image_path, settings)

print("Image shape:", image_analysis.image.shape)
print("Pixel size:", image_analysis.pixel_size)
print("Number of detections:", len(image_analysis.detections))
print("Number of GUV objects:", len(image_analysis.guvs))


#%%
# 2. Calculate image-level background

image_analysis.calculate_global_background()

print("Global background:", image_analysis.global_background)
print("Image-view global background:", image_analysis.image_view.global_background)
print("Global background check passed.")


#%%
# 3. Select one GUV and check basic properties

guv = image_analysis.guvs[guv_id_to_test]

print("GUV ID:", guv.id)
print("Coordinates:", guv.ves_coordinates)
print("Number of angles:", guv.num_angles)
print("Along-radius shape:", guv.along_radius.shape)
print("Theta shape:", guv.theta.shape)
print("GUV object check passed.")


#%%
# 4. Check intensity profiles and background correction

guv.calculate_intensity_profiles()

print("Intensity profile shape:", guv.analysis.intensity_profiles.shape)
print("Death mark after profile calculation:", guv.death_mark)
print("Comments:", guv.analysis.comments)

guv.set_background()

print("Background method:", guv.analysis.background_method)
print("Background:", guv.analysis.background)

guv.correct_intensity_profiles()

print("Profile and background-correction check passed.")


#%%
# 5. Run one analysis route

if analysis_route == "circular":
    guv.run_circular_analysis()
    print("Circular analysis completed.")
    print("Circular membrane fraction:", guv.analysis.circular_fraction_membrane)
    print("Circular localization:", guv.analysis.circular_memb_localization)
    print("Circular inside intensity:", guv.analysis.circular_inside_intensity)

elif analysis_route == "noncircular":
    guv.run_noncircular_analysis()

    print("Non-circular analysis completed.")
    print("Non-circular membrane fraction:", guv.analysis.noncircular_fraction_membrane)
    print("Non-circular localization:", guv.analysis.noncircular_memb_localization)
    print("Non-circular inside intensity:", guv.analysis.noncircular_inside_intensity)

else:
    raise ValueError("analysis_route must be 'circular' or 'noncircular'.")

print("Death mark:", guv.death_mark)
print("Comments:", guv.analysis.comments)


#%%
# 6. Plot the selected GUV

if settings.plotting.show and not guv.death_mark:
    if analysis_route == "circular":
        guv.plot_circular_shape()
        guv.plot_circular_profiles()

    if analysis_route == "noncircular":
        guv.plot_noncircular_shape()
        guv.plot_noncircular_profiles()


#%%
# 7. Run the same route for all GUVs in the image

for guv_id, guv in image_analysis.guvs.items():
    if analysis_route == "circular":
        guv.run_circular_analysis()

    if analysis_route == "noncircular":
        guv.run_noncircular_analysis()

image_analysis.kill_guvs_on_comment()

print("Good GUVs:", len(image_analysis.good_GUVs))
print("Bad GUVs:", len(image_analysis.bad_GUVs))


#%%
# 8. Save and reload results

if save_results:
    image_root, image_ext = os.path.splitext(image_path)
    output_path = image_root + f"_{analysis_route}_analysis.h5"

    image_analysis.save_hdf5(output_path, guvs="all")
    print("Saved:", output_path)

    reloaded_image = load_hdf5(output_path)

    print("Reloaded good GUVs:", len(reloaded_image.good_GUVs))
    print("Reloaded bad GUVs:", len(reloaded_image.bad_GUVs))

    if analysis_route == "circular":
        values = reloaded_image.extract_parameter(
            "analysis.circular_inside_intensity",
            GUVs="good"
        )

    if analysis_route == "noncircular":
        values = reloaded_image.extract_parameter(
            "analysis.noncircular_inside_intensity",
            GUVs="good"
        )

    print("Extracted inside intensity for", len(values), "good GUVs.")
    print("HDF5 save/load check passed.")


#%%
# 9. Optional quick result plot

if save_results:
    inside_intensity = values

    ch0 = [val[0] for val in inside_intensity.values() if val is not None]
    ch1 = [val[1] for val in inside_intensity.values() if val is not None and len(val) > 1]

    if len(ch0) > 0:
        plt.figure()
        plt.hist(ch0, bins=30)
        plt.xlabel("Inside intensity channel 0")
        plt.ylabel("Count")
        plt.title(f"{analysis_route} route")
        plt.show()

    if len(ch0) > 0 and len(ch1) > 0:
        plt.figure()
        plt.scatter(ch0, ch1)
        plt.xlabel("Inside intensity channel 0")
        plt.ylabel("Inside intensity channel 1")
        plt.title(f"{analysis_route} route")
        plt.show()