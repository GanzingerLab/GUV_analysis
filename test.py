#%%
from guv_analysis.settings import AnalysisSettings
from guv_analysis.image import GUVImage
from guv_analysis.hdf_manager import load_hdf5
import matplotlib.pyplot as plt
import numpy as np
import os
from glob import glob
from tqdm import tqdm 
import pandas as pd
import importlib
import guv_analysis.hdf_manager as hdf_manager

#%%
path = r'\\sun.amolf.nl\ganzinger\project-folder\26 Optimizing PURE encapsulation and function in GUV Franzi\analysis\disguvery\testing\20260702_Exp_36_wellscans_N5_19H_1024px_#01_cropped_tiffs\test'
images = glob(os.path.join(path, "**", "*"+'.tif'), recursive=True)
settings = AnalysisSettings()
settings.profiles.length_excess=3
settings.guv_ch = 1
settings.background.method = 'global'
settings.profiles.noncircular_membrane_width_pixels = 5
results = {}
for image in images:
    image_analysis = GUVImage(image, settings)
    image_analysis.filter_GUVs_in_clusters()
    image_analysis.calculate_global_background()
    for guv_id, guv in image_analysis.guvs.items():
        guv.run_circular_analysis()
        guv.run_noncircular_analysis()
    image_analysis.kill_guvs_on_comment()
    image_analysis.save_hdf5(image.replace('.tif', '_analysis.h5'), guvs="all")
#%%
guv = image_analysis.guvs[3]
membrane_locs = guv.analysis.membrane.peak_radius_by_angle
along_radius = guv.along_radius
#%%
for i in list(image_analysis.good_GUVs):
    guv = image_analysis.guvs[i]
    guv.plot_noncircular_profiles()

#%%
id = 3
# image2 = load_hdf5(r'\\sun.amolf.nl\ganzinger\project-folder\26 Optimizing PURE encapsulation and function in GUV Franzi\analysis\disguvery\testing\20260702_Exp_36_wellscans_N5_19H_1024px_#01_cropped_tiffs\test\tile_r01_c01_analysis.h5')
#%%
ints = image2.extract_parameter("analysis.noncircular_inside_intensity", GUVs="good")
print(ints)
int_ch0 = [val[0] for val in ints.values() if val is not None]
int_ch1 = [val[1] for val in ints.values() if val is not None]
image2.guvs[id].plot_noncircular_profiles()
print(image2.guvs[id].analysis.noncircular_inside_intensity)
plt.scatter(int_ch0, int_ch1)
plt.show()
plt.hist(int_ch1, bins=50)

#%%
for guv_id in list(image2.good_GUVs):
    guv = image2.guvs[guv_id]
    guv.plot_noncircular_shape()
    guv.plot_circular_shape()
    print(guv.analysis.comments)
#%%
print(image_analysis.pixel_size)
sizes = image2.extract_parameter("analysis.membrane.mean_radius", GUVs="good")
sizes = {k: v/image2.pixel_size for k, v in sizes.items() if v is not None}
plt.hist(sizes.values(), bins=50)
# plt.ylim(0, 50000)
# guvs_to_test = [1, 8, 2.0, 21.0, 27.0, 74.0, 16, 77.0, 81.0, 82.0, 91.0, 102.0, 103.0, 106.0, 108.0, 112.0, 22, 113.0, 122.0, 123.0, 128.0, 133.0, 134.0, 136.0, 140.0, 143.0, 144.0, 145.0, 180, 148.0, 149.0, 150.0, 154.0, 155.0, 156.0, 168.0, 173.0, 179.0, 181.0, 183.0, 184.0, 185.0, 189.0, 190.0, 191.0, 196.0, 201.0, 202.0, 203.0, 209.0, 211.0, 212.0, 213.0, 216.0, 218.0, 219.0, 220.0, 226.0, 227.0, 228.0, 229.0, 231.0, 237.0, 238.0, 239.0, 243.0, 244.0, 245.0, 251.0, 252.0, 258.0, 262.0, 265.0, 266.0, 267.0, 268.0, 269.0, 274.0, 277.0, 278.0, 279.0, 280.0, 287.0, 290.0, 293.0, 296.0, 297.0, 299.0, 300.0, 301.0, 302.0, 303.0, 304.0, 305.0, 307.0, 308.0, 311.0, 312.0, 313.0, 314.0, 315.0, 316.0, 317.0, 318.0, 323.0, 324.0, 325.0, 326.0, 327.0, 329.0, 331.0, 332.0, 333.0, 334.0, 335.0, 336.0, 338.0, 339.0, 340.0, 341.0, 342.0, 344.0, 347.0, 348.0, 349.0, 350.0, 351.0, 353.0, 361.0, 363.0, 364.0, 365.0, 366.0, 369.0, 370.0, 371.0, 372.0, 374.0, 376.0, 377.0, 378.0, 380.0, 382.0, 384.0, 385.0, 386.0, 389.0, 390.0, 391.0, 393.0, 394.0, 396.0, 397.0, 398.0, 400.0, 401.0, 402.0, 404.0, 405.0, 409.0, 411.0, 414.0, 416.0, 417.0, 418.0, 419.0, 420.0, 421.0, 425.0, 428.0, 429.0, 430.0, 433.0, 434.0, 435.0, 436.0, 437.0, 438.0, 440.0, 441.0, 442.0, 444.0, 446.0, 448.0, 449.0, 450.0, 451.0, 452.0, 453.0, 454.0, 457.0, 460.0, 461.0, 464.0, 465.0, 466.0, 467.0, 468.0, 469.0, 471.0, 472.0, 473.0, 477.0, 478.0, 479.0, 483.0, 484.0, 485.0, 486.0, 487.0, 488.0, 490.0, 491.0, 494.0, 498.0, 499.0, 501.0, 507.0, 508.0, 509.0, 510.0, 511.0, 515.0, 517.0, 519.0, 520.0, 521.0, 522.0, 526.0, 527.0, 529.0, 533.0]
# common_guvs = sorted(set(bad_guvs) & set(guvs_to_test))
# print(common_guvs)
# guvs_to_test = [int(i) for i in guvs_to_test]
# guvs_to_test = image_analysis.good_GUVs.copy()
# for i in guvs_to_test:
#     guv = guvs[i]
#     guv.run_noncircular_analysis()
    # guv.plot_noncircular_shape()

    # print("GUV:", i)
    # print("Membrane support fraction:", guv.analysis.noncircular_fraction_membrane)
    # print("Comments:", guv.analysis.comments)
    # print("Death mark:", guv.death_mark)
#%%
image_analysis.kill_guvs_on_comment()
bad_guvs2 = image_analysis.bad_GUVs
print(bad_guvs2)
print(len(image_analysis.good_GUVs))
print(len(bad_guvs2))
#%%
guvs_to_test = image_analysis.good_GUVs.copy()
for i in guvs_to_test:
    guv = guvs[i]
    # guv.run_noncircular_analysis()
    guv.plot_noncircular_shape()
#%%
profiles = {}

for i in image_analysis.bad_GUVs:
    guv = image_analysis.guvs[i]
    try:
        guv.plot_noncircular_shape()
    except:
        x = int(round(guv.xc))
        y = int(round(guv.yc))
        r = int(round(guv.radius))
        plt.imshow(guv.image_view.image[guv.settings.guv_ch, 
                    y-r:y+r, 
                    x-r:x+r], cmap='gray')
        plt.title(f"GUV {guv.id}")

#%%
id = 25
print(image_analysis.guvs[id].death_mark)
print(image_analysis.guvs[id].analysis.comments)
print(image_analysis.guvs[id].analysis.noncircular_fraction_membrane)
image_analysis.guvs[id].plot_noncircular_shape()
image_analysis.guvs[id].plot_noncircular_profiles()
#%%
path = r'\\sun.amolf.nl\ganzinger\project-folder\26 Optimizing PURE encapsulation and function in GUV Franzi\analysis\disguvery\testing\20260702_Exp_36_wellscans_N5_19H_1024px_#01_cropped_tiffs\test'
images = glob(os.path.join(path, "**", "*"+'.tif'), recursive=True)
settings = AnalysisSettings()
settings.profiles.length_excess=2
settings.guv_ch = 1
settings.background.method = 'global'
settings.profiles.noncircular_membrane_width_pixels = 5
results = {}
for image in images:
    image_analysis = GUVImage(image, settings)
    image_analysis.filter_GUVs_in_clusters()
    image_analysis.calculate_global_background()
guvs = image_analysis.guvs
print(image_analysis.bad_GUVs)

guvs_to_test = [1, 8, 2.0, 21.0, 27.0, 74.0, 16, 77.0, 81.0, 82.0, 91.0, 102.0, 103.0, 106.0, 108.0, 112.0, 22, 113.0, 122.0, 123.0, 128.0, 133.0, 134.0, 136.0, 140.0, 143.0, 144.0, 145.0, 180,148.0, 149.0, 150.0, 154.0, 155.0, 156.0, 168.0, 173.0, 179.0, 181.0, 183.0, 184.0, 185.0, 189.0, 190.0, 191.0, 196.0, 201.0, 202.0, 203.0, 209.0, 211.0, 212.0, 213.0, 216.0, 218.0, 219.0, 220.0, 226.0, 227.0, 228.0, 229.0, 231.0, 237.0, 238.0, 239.0, 243.0, 244.0, 245.0, 251.0, 252.0, 258.0, 262.0, 265.0, 266.0, 267.0, 268.0, 269.0, 274.0, 277.0, 278.0, 279.0, 280.0, 287.0, 290.0, 293.0, 296.0, 297.0, 299.0, 300.0, 301.0, 302.0, 303.0, 304.0, 305.0, 307.0, 308.0, 311.0, 312.0, 313.0, 314.0, 315.0, 316.0, 317.0, 318.0, 323.0, 324.0, 325.0, 326.0, 327.0, 329.0, 331.0, 332.0, 333.0, 334.0, 335.0, 336.0, 338.0, 339.0, 340.0, 341.0, 342.0, 344.0, 347.0, 348.0, 349.0, 350.0, 351.0, 353.0, 361.0, 363.0, 364.0, 365.0, 366.0, 369.0, 370.0, 371.0, 372.0, 374.0, 376.0, 377.0, 378.0, 380.0, 382.0, 384.0, 385.0, 386.0, 389.0, 390.0, 391.0, 393.0, 394.0, 396.0, 397.0, 398.0, 400.0, 401.0, 402.0, 404.0, 405.0, 409.0, 411.0, 414.0, 416.0, 417.0, 418.0, 419.0, 420.0, 421.0, 425.0, 428.0, 429.0, 430.0, 433.0, 434.0, 435.0, 436.0, 437.0, 438.0, 440.0, 441.0, 442.0, 444.0, 446.0, 448.0, 449.0, 450.0, 451.0, 452.0, 453.0, 454.0, 457.0, 460.0, 461.0, 464.0, 465.0, 466.0, 467.0, 468.0, 469.0, 471.0, 472.0, 473.0, 477.0, 478.0, 479.0, 483.0, 484.0, 485.0, 486.0, 487.0, 488.0, 490.0, 491.0, 494.0, 498.0, 499.0, 501.0, 507.0, 508.0, 509.0, 510.0, 511.0, 515.0, 517.0, 519.0, 520.0, 521.0, 522.0, 526.0, 527.0, 529.0, 533.0]
guvs_to_test = [int(i) for i in guvs_to_test]
for i in guvs_to_test[0:30]:
    guv = guvs[i]
    guv.run_circular_analysis()
    guv.detect_noncircular_membrane()
    # guv.calculate_normalized_noncircular_radial_profile()
    guv.calculate_noncircular_angular_profile()
    guv.plot_noncircular_shape()
    angular_profile = guv.analysis.noncircular_angular_profiles[:, 1]
    theta = guv.theta

    flattened_profile, fitted_profile, death_mark, comments = flatten_angular_profile(angular_profile, theta)
    print(death_mark)
    print(comments)
    results[i] = [comments, death_mark]
    smoothed_profile = circular_rolling_average(angular_profile, 5)
    smoothed_flattened = circular_rolling_average(flattened_profile, 5)
    
    # Normalize only for visual comparison
    profile_for_plot = smoothed_profile / np.nanmedian(smoothed_profile)
    fit_for_plot = fitted_profile / np.nanmedian(fitted_profile)

    # Keep the flattened profile centered around 1
    flattened_for_plot = smoothed_flattened / np.nanmedian(smoothed_flattened)

    plt.figure(figsize=(9, 4))
    plt.plot(theta, profile_for_plot, color="blue", label="Angular profile")
    plt.plot(theta, fit_for_plot, color="gray", label="Polarization fit")
    plt.plot(theta, flattened_for_plot, color="orange", label="Flattened profile")
    plt.axhline(1, color="black", linestyle="--", linewidth=1, label="Expected flattened signal")
    plt.xlabel("Angle (rad)")
    plt.ylabel("Normalized intensity")
    plt.legend()
    plt.tight_layout()
    plt.show()
results_df = pd.DataFrame.from_dict(
    results,
    orient="index",
    columns=["comments", "death_mark"],
)

results_df.index.name = "guv_index"

#%%
from scipy.ndimage import median_filter

def remove_broad_angular_variation(flattened_profile, window_degrees=45):
    """Remove broad membrane-intensity regions from a flattened angular profile."""
    flattened_profile = np.asarray(flattened_profile, dtype=float)
    angle_step = 360 / flattened_profile.size
    window = max(3, int(round(window_degrees / angle_step)))

    if window % 2 == 0:
        window += 1

    broad_profile = median_filter(flattened_profile, size=window, mode="wrap")
    broad_profile = np.maximum(broad_profile, np.finfo(float).eps)
    local_profile = flattened_profile / broad_profile
    return local_profile, broad_profile

angular_profile = guv.analysis.noncircular_angular_profiles[:, 1]
flattened_profile, fitted_profile = flatten_angular_profile(angular_profile, guv.theta)

local_profile, broad_profile = remove_broad_angular_variation(
    flattened_profile,
    window_degrees=45,
)

smoothed_angular = circular_rolling_average(angular_profile, 5)
smoothed_flattened = circular_rolling_average(flattened_profile, 5)
smoothed_local = circular_rolling_average(local_profile, 5)

angular_plot = smoothed_angular / np.nanmedian(smoothed_angular)
fit_plot = fitted_profile / np.nanmedian(fitted_profile)
flattened_plot = smoothed_flattened / np.nanmedian(smoothed_flattened)
broad_plot = broad_profile / np.nanmedian(broad_profile)
local_plot = smoothed_local / np.nanmedian(smoothed_local)

plt.figure(figsize=(10, 4))
plt.plot(guv.theta, angular_plot, color="blue", label="Angular profile")
plt.plot(guv.theta, fit_plot, color="gray", label="Polarization fit")
plt.plot(guv.theta, flattened_plot, color="orange", label="Polarization corrected")
plt.plot(guv.theta, broad_plot, color="green", label="Broad intensity variation")
plt.axhline(1, color="black", linestyle="--")
plt.xlabel("Angle (rad)")
plt.ylabel("Median-normalized intensity")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(guv.theta, local_plot, color="orange", label="Local residual profile")
plt.axhline(1, color="black", linestyle="--", label="Expected local signal")
plt.xlabel("Angle (rad)")
plt.ylabel("Intensity relative to local membrane")
plt.legend()
plt.tight_layout()
plt.show()
#%%
path = r'\\sun.amolf.nl\ganzinger\project-folder\26 Optimizing PURE encapsulation and function in GUV Franzi\analysis\disguvery\testing\20260702_Exp_36_wellscans_N5_19H_1024px_#01_cropped_tiffs\test'
images = glob(os.path.join(path, "**", "*"+'.tif'), recursive=True)
settings = AnalysisSettings()
settings.guv_ch = 1
settings.background.method = 'global'
for image in images:
    image_analysis = GUVImage(image, settings)

    # Image-level calculations
    image_analysis.calculate_global_background()
    for idx, guv1 in tqdm(image_analysis.guvs.items()):
        # Select one GUV
        # guv1 = image_analysis.guvs[1]

        guv1.run_circular_analysis()
        guv1.run_noncircular_analysis()

        # Show results
        # guv1.plot_circular_shape()
        # guv1.plot_circular_profiles()
        # guv1.plot_noncircular_shape()
        # guv1.plot_noncircular_profiles()

        # Inspect stored results
        # print("GUV ID:", guv1.id)
        # print("Comments:", guv1.analysis.comments)
        # print("Death mark:", guv1.death_mark)

        # print("\nCircular membrane:")
        # print("Peak index:", guv1.analysis.membrane.peak_index)
        # print("Peak radius:", guv1.analysis.membrane.peak_radius)
        # print("Inner border index:", guv1.analysis.membrane.inner_border_index)
        # print("Outer border index:", guv1.analysis.membrane.outer_border_index)

        # print("\nCircular quantification:")
        # print("Localization:", guv1.analysis.circular_memb_localization)
        # print("Inside intensity:", guv1.analysis.circular_inside_intensity)

        # print("\nNoncircular membrane:")
        # print("Peak radius by angle:", guv1.analysis.membrane.peak_radius_by_angle)
        # print("Shape x:", guv1.analysis.membrane.shape_x)
        # print("Shape y:", guv1.analysis.membrane.shape_y)

        # print("\nNoncircular quantification:")
        # print("Localization:", guv1.analysis.noncircular_memb_localization)
        # print("Inside intensity:", guv1.analysis.noncircular_inside_intensity)
#%%
path = r'D:\Data\EVOLF\20260617_GUV_FLuo4_TRIMEB_aHL_perm_2_test_analysis_radial_profile\20260610_D4_t0-04.czi'

settings = AnalysisSettings()
image_analysis = GUVImage(path, settings)

# 1. Check image loading
print("Image shape:", image_analysis.image.shape)
print("Pixel size:", image_analysis.pixel_size)
print("Number of detections:", len(image_analysis.detections))
print("Number of GUV objects:", len(image_analysis.guvs))

assert image_analysis.image is not None
assert image_analysis.pixel_size > 0
assert len(image_analysis.guvs) > 0


# 2. Check global mask and global background
image_analysis.calculate_global_background()

print("Global mask:", image_analysis.global_mask)
print("Global background:", image_analysis.global_background)
print(
    "View global background:",
    image_analysis.image_view.global_background,
)

assert image_analysis.global_mask is not None
assert image_analysis.global_background is not None
assert image_analysis.image_view.global_background is not None
assert np.allclose(
    image_analysis.global_background,
    image_analysis.image_view.global_background,
)


# 3. Select one GUV
guv1 = image_analysis.guvs[2]

print("GUV:", guv1)
print("Coordinates:", guv1.ves_coordinates)
print("Number of angles:", guv1.num_angles)
print("Along-radius shape:", guv1.along_radius.shape)
print("Theta shape:", guv1.theta.shape)

assert guv1.ves_coordinates.shape == (4,)
assert guv1.num_angles == len(guv1.theta)
assert len(guv1.along_radius) > 0


# 4. Check intensity-profile calculation
guv1.calculate_intensity_profiles()

print(
    "intensity profile shape:",
    guv1.analysis.intensity_profiles.shape,
)
print("Death mark:", guv1.death_mark)
print("Comments:", guv1.analysis.comments)

assert guv1.analysis.intensity_profiles is not None
assert guv1.analysis.intensity_profiles.ndim == 3
assert guv1.analysis.intensity_profiles.shape[0] == len(guv1.along_radius)
assert guv1.analysis.intensity_profiles.shape[1] == guv1.num_angles


# 5. Check background selection
guv1.set_background()

print("Background method:", guv1.analysis.background_method)
print("Background:", guv1.analysis.background)
print("Comments:", guv1.analysis.comments)
print("Death mark:", guv1.death_mark)

assert guv1.analysis.background_method in ("local", "global")
assert guv1.analysis.background is not None
assert guv1.analysis.background.shape[0] == image_analysis.image.shape[0]


# 6. Check corrected profiles
guv1.correct_intensity_profiles()

print(
    "Corrected profile shape:",
    guv1.analysis.intensity_profiles.shape,
)

assert guv1.analysis.intensity_profiles is not None

guv1.calculate_circular_radial_profile()

print("All checks passed.")
