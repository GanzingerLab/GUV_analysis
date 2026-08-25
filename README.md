# GUV analysis

Early testing version of a Python package for analyzing giant unilamellar vesicles (GUVs) from microscopy images.

The package starts from an image and a matching detection CSV file, created with disGUVery. It creates one `GUVImage` containing one `GUV` object per detected vesicle, extracts intensity profiles, detects the membrane, calculates membrane localization and inside intensity, filters bad GUVs, and saves the results to HDF5.

This code is meant to follow **one analysis route at a time**:

- circular analysis, for approximately circular GUVs --> using disGUVery detection.
- non-circular analysis, for deformed or non-circular GUVs --> refines detection using dynamic programming. 

Running both routes on the same GUV is useful for testing, comparison, and debugging, but it is not the recommended final analysis workflow.

---
## Installation

1. Clone or copy this repository to your computer.

2. Open the Anaconda Prompt in the main package folder and run:

   ```bash
   conda env create -f environment.yml
   ```
---

## 1. Expected input files
For every image, the package expects a matching detection CSV file in the same folder.

Example:

```text
sample.tif
sample_detected_vesicles.csv
```

or:

```text
sample.czi
sample_detected_vesicles.csv
```

The detection CSV is expected to contain one GUV per row:

```text
vesicle_id, xc, yc, radius
```

where `xc` and `yc` are the GUV center coordinates in pixels, and `radius` is the approximate radius in pixels.

The detection suffix can be changed with:

```python
settings.detection_suffix = "_detected_vesicles.csv"
```

---

## 2. Basic imports

```python
from guv_analysis.settings import AnalysisSettings
from guv_analysis.image import GUVImage
from guv_analysis.hdf_manager import load_hdf5

import os
from glob import glob
import matplotlib.pyplot as plt
```
---

## 3. Create settings

```python
settings = AnalysisSettings() 
```
You can go to the settings file in guv_analysis/settings.py. --> Please, never commit thise changes to github. 

you can also modify some setting in the running code. For exmaple: 

Set the membrane channel. Channel numbering starts at 0.

```python
settings.guv_ch = 1
```

Choose the background method.

```python
settings.background.method = "global"
```

Available options are:

```text
"local"
"global"
"local_then_global"
```

Some useful profile settings:

To print all current settings:

```python
settings.show()
```

---

## 4. Package structure: main classes and files

The package is organized around two main objects: `GUVImage` for one complete image, and `GUV` for one vesicle inside that image. 

### `GUVImage`

`GUVImage` represents one image and all GUVs detected in that image.

```python
image_analysis = GUVImage(image_path, settings)
```

It contains:

```text
image_analysis.image            image array
image_analysis.pixel_size       pixel size read from the image metadata. If not avilable defaults to 1.0. 
image_analysis.detections       detection table loaded from the disGUVery CSV
image_analysis.guvs             dictionary with one GUV object per detected vesicle. 
image_analysis.good_GUVs        IDs of GUVs currently considered good
image_analysis.bad_GUVs         IDs of GUVs marked as bad
```

Important methods are:

```text
calculate_global_background()   calculate image-level background
filter_GUVs_in_clusters()       mark clustered/overlapping GUVs as bad
kill_guvs_on_comment()          move GUVs with killing comments to bad_GUVs
extract_parameter()             collect one stored parameter from many GUVs
save_hdf5()                     save the image analysis and GUV results
```

### `GUV`

Each entry in `image_analysis.guvs` is a `GUV` object. A `GUV` object represents one detected vesicle.

```python
guv = image_analysis.guvs[3] #the Id passed, in this case 3, corresponds to the ID assigned by disGUVery. 
```

It contains the original detection information:

```text
guv.id                          vesicle ID from the detection CSV
guv.xc, guv.yc                  vesicle center coordinates in pixels
guv.radius                      approximate radius from the detection CSV
guv.death_mark                  True if the GUV failed a quality-control step
guv.analysis                    object containing calculated results
```

The two main analysis routes are:

```python
guv.run_circular_analysis()
```

or:

```python
guv.run_noncircular_analysis()
```

For final analysis, choose one route for a dataset. Running both routes on the same GUV is mainly useful for comparison, testing, or debugging.

Useful plotting methods are:

```text
plot_circular_shape()           show the circular membrane detection
plot_noncircular_shape()        show the non-circular membrane contour
plot_circular_profiles()        show circular radial/angular profiles
plot_noncircular_profiles()     show non-circular radial/angular profiles
```

Results are saves within the analysis object within the GUV. it can be accessed as guv.analysis. Membrane detection results are saved within a different object within analysis. It can be accessed as guv.analysis.membrane. Examples:

```text
analysis.comments                           comments and warnings for this GUV
analysis.circular_fraction_membrane         circular membrane support fraction
analysis.noncircular_fraction_membrane      non-circular membrane support fraction
analysis.circular_memb_localization         circular membrane-localization values
analysis.noncircular_memb_localization      non-circular membrane-localization values
analysis.circular_inside_intensity          circular inside-intensity values
analysis.noncircular_inside_intensity       non-circular inside-intensity values
analysis.membrane.mean_radius               detected mean membrane radius
analysis.membrane.shape_x                   non-circular membrane x coordinates
analysis.membrane.shape_y                   non-circular membrane y coordinates
```

To see all avilable parameters check directly the object in guv_analysis/GUV.py and look inside class GUVAnalysis: and class MembraneDetectionResult:. 

### Helper modules

Most users do not need to call the helper modules directly, but they contain the lower-level functions used by `GUV` and `GUVImage`.

```text
io_tools.py                 image and detection loading helpers
background.py               local/global background estimation
profiles.py                 radial and angular profile extraction
membrane_detection.py       circular and non-circular membrane detection
signal_quantification.py    localization and inside-intensity calculations
filter.py                   membrane-fraction and cluster filters
plotting.py                 plotting functions
hdf_manager.py              HDF5 saving and loading
```
Fill analysis can be also done importing those functions. The initialized objects only contain wrappers to properly run and store the output of the functions. 
---

## 5. Analyze one image with the non-circular route

Use this route when GUVs may be deformed or not perfectly circular. 

```python
image_path = r"CHANGE_THIS_TO_ONE_IMAGE.tif" #path to the image

settings = AnalysisSettings() #initialize settings

image_analysis = GUVImage(image_path, settings) #initialize GUVImage object

image_analysis.filter_GUVs_in_clusters() #Filter GUVs in large clusters
image_analysis.calculate_global_background() #Calculates the global background from the mask used to filter the GUVs in clusters

for guv_id, guv in image_analysis.guvs.items(): #for each GUV
    guv.run_noncircular_analysis() #run the full analysis pipleine

image_analysis.kill_guvs_on_comment() #kill GUVs based on the comments produced during the analysis. 

output_path = os.path.splitext(image_path)[0] + "_analysis.h5" #make path for the ouptut file 
image_analysis.save_hdf5(output_path, guvs="all") #save output file. This saves everything and can be used to reinitilize the object in the state you left it in. 
```

---

## 6. Analyze one image with the circular route

Use this route when the circular membrane approximation is sufficient. Comments are the same as above, but is uses run_circular_analysis(). 

```python
image_path = r"CHANGE_THIS_TO_ONE_IMAGE.tif"

settings = AnalysisSettings()

image_analysis = GUVImage(image_path, settings)

image_analysis.filter_GUVs_in_clusters()
image_analysis.calculate_global_background()

for guv_id, guv in image_analysis.guvs.items():
    guv.run_circular_analysis()

image_analysis.kill_guvs_on_comment()

output_path = os.path.splitext(image_path)[0] + "_analysis.h5"
image_analysis.save_hdf5(output_path, guvs="all")
```

---

## 7. Analyze a full folder

Choose either `run_noncircular_analysis()` or `run_circular_analysis()`. Do not use both for the final analysis of the same dataset unless you are explicitly comparing methods.

### Non-circular batch analysis

```python
path = r"CHANGE_THIS_TO_YOUR_IMAGE_FOLDER"
images = glob(os.path.join(path, "**", "*.tif"), recursive=True) #looks for all tif (or czi) files, and runs them. 

settings = AnalysisSettings()


for image_path in images:
    image_analysis = GUVImage(image_path, settings)

    image_analysis.filter_GUVs_in_clusters()
    image_analysis.calculate_global_background()

    for guv_id, guv in image_analysis.guvs.items():
        guv.run_noncircular_analysis()

    image_analysis.kill_guvs_on_comment()

    output_path = os.path.splitext(image_path)[0] + "_analysis.h5"
    image_analysis.save_hdf5(output_path, guvs="all")
```

For `.czi` files, change the image search line:

```python
images = glob(os.path.join(path, "**", "*.czi"), recursive=True)
```

---

## 8. Inspect one GUV before running a full batch

It is recommended to test one image and one GUV first.

```python
image_path = r"CHANGE_THIS_TO_ONE_IMAGE.tif"

settings = AnalysisSettings()
settings.guv_ch = 1
settings.background.method = "global"
settings.plotting.show = True

image_analysis = GUVImage(image_path, settings)
image_analysis.calculate_global_background()

print("Image shape:", image_analysis.image.shape)
print("Pixel size:", image_analysis.pixel_size)
print("Number of detections:", len(image_analysis.detections))
print("Number of GUVs:", len(image_analysis.guvs))

guv = image_analysis.guvs[1]
guv.run_noncircular_analysis()

guv.plot_noncircular_shape()
guv.plot_noncircular_profiles()

print("Comments:", guv.analysis.comments)
print("Death mark:", guv.death_mark)
```

For circular analysis, replace:

```python
guv.run_noncircular_analysis()
guv.plot_noncircular_shape()
guv.plot_noncircular_profiles()
```

with:

```python
guv.run_circular_analysis()
guv.plot_circular_shape()
guv.plot_circular_profiles()
```

---

## 9. Plot results

After running the selected analysis route, plots can be shown for individual GUVs.

### Non-circular route

```python
guv = image_analysis.guvs[1]

guv.plot_noncircular_shape()
guv.plot_noncircular_profiles()
```

To save separate radial and angular profile plots:

```python
guv.save_noncircular_profile_plots(r"CHANGE_THIS_TO_OUTPUT_FOLDER")
```

### Circular route

```python
guv = image_analysis.guvs[1]

guv.plot_circular_shape()
guv.plot_circular_profiles()
```

To save separate radial and angular profile plots:

```python
guv.save_circular_profile_plots(r"CHANGE_THIS_TO_OUTPUT_FOLDER")
```

---

## 10. Good and bad GUVs

The package tracks good and bad GUV IDs:

```python
print("Good GUVs:", image_analysis.good_GUVs)
print("Bad GUVs:", image_analysis.bad_GUVs)
```

Some GUVs are marked dead directly during analysis. Others may only receive comments. To kill GUVs based on the current filter settings:

```python
image_analysis.kill_guvs_on_comment()
```

The killing comments are defined in:

```python
settings.filter.killing_comments
```

The warning comments that do not kill a GUV are defined in:

```python
settings.filter.surviving_comments
```

To inspect one GUV:

```python
guv = image_analysis.guvs[1]

print("Death mark:", guv.death_mark)
print("Comments:", guv.analysis.comments)
```

---

## 11. Save results

Save all GUVs:

```python
image_analysis.save_hdf5("analysis.h5", guvs="all")
```

Save only good GUVs:

```python
image_analysis.save_hdf5("analysis_good.h5", guvs="good")
```

Save only bad GUVs:

```python
image_analysis.save_hdf5("analysis_bad.h5", guvs="bad")
```

You can also save selected GUV IDs:

```python
image_analysis.save_hdf5("analysis_selected.h5", guvs=[1, 3, 8])
```

---

## 12. Load saved results

```python
image_analysis = load_hdf5(r"CHANGE_THIS_TO_ANALYSIS_FILE.h5")
```

If the original image moved since the HDF5 file was created, use:

```python
image_analysis = load_hdf5(
    r"CHANGE_THIS_TO_ANALYSIS_FILE.h5",
    path_override=r"CHANGE_THIS_TO_ORIGINAL_IMAGE.tif"
)
```

Then inspect or plot as usual:

```python
print("Good GUVs:", image_analysis.good_GUVs)
print("Bad GUVs:", image_analysis.bad_GUVs)

guv = image_analysis.guvs[1]
guv.plot_noncircular_shape()
guv.plot_noncircular_profiles()
```

---

## 13. Extract parameters

Use `extract_parameter()` to collect one stored value from many GUVs.

Example: extract non-circular inside intensity from good GUVs.

```python
inside_intensity = image_analysis.extract_parameter(
    "analysis.noncircular_inside_intensity",
    GUVs="good"
)
```

This returns a dictionary:

```python
{
    guv_id: value,
    ...
}
```

Example plot for two channels:

```python
int_ch0 = [val[0] for val in inside_intensity.values() if val is not None]
int_ch1 = [val[1] for val in inside_intensity.values() if val is not None]

plt.scatter(int_ch0, int_ch1)
plt.xlabel("Channel 0 inside intensity")
plt.ylabel("Channel 1 inside intensity")
plt.show()
```

Example: extract mean detected membrane radius.

```python
sizes = image_analysis.extract_parameter(
    "analysis.membrane.mean_radius",
    GUVs="good"
)
```

Convert from pixels to micrometers if `pixel_size` is in micrometers per pixel:

```python
sizes_um = {
    guv_id: radius * image_analysis.pixel_size
    for guv_id, radius in sizes.items()
    if radius is not None
}

plt.hist(sizes_um.values(), bins=50)
plt.xlabel("GUV radius (um)")
plt.ylabel("Count")
plt.show()
```

---

## 14. Useful stored parameters

Most calculated results are stored inside `guv.analysis`.

For a detailed explanation of each output, see:

[`analysis_outputs.md`](analysis_outputs.md)

Depending on the route you used, useful parameters include:

### General

```text
analysis.comments
analysis.background
analysis.background_method
analysis.membrane.mean_radius
```

### Circular route

```text
analysis.circular_radial_profiles
analysis.circular_angular_profiles
analysis.circular_memb_localization
analysis.circular_inside_intensity
analysis.circular_fraction_membrane
analysis.membrane.peak_index
analysis.membrane.peak_radius
analysis.membrane.inner_border_index
analysis.membrane.outer_border_index
```

### Non-circular route

```text
analysis.noncircular_radial_profiles
analysis.noncircular_angular_profiles
analysis.noncircular_memb_localization
analysis.noncircular_inside_intensity
analysis.noncircular_fraction_membrane
analysis.membrane.peak_radius_by_angle
analysis.membrane.shape_x
analysis.membrane.shape_y
```
---

## 15. Notes 

This is an early testing version. Useful feedback includes:

- Does the package run on your images?
- Are the detected contours reasonable?
- Are too many or too few GUVs marked as bad?
- Are the comments useful?
- Are the settings understandable?
- Are the HDF5 files easy to reload?
- Which plots are missing or confusing?

Please report both crashes and scientifically suspicious results.

---

## 16. Common problems

### Detection CSV not found

Check that the CSV is in the same folder as the image and has the expected suffix:

```python
settings.detection_suffix = "_detected_vesicles.csv"
```

### Global background not calculated

If using:

```python
settings.background.method = "global"
```

or:

```python
settings.background.method = "local_then_global"
```

then run:

```python
image_analysis.calculate_global_background()
```

before running GUV analysis.

### Plots keep popping up during batch analysis

Use:

```python
settings.plotting.show = False
```

and avoid calling plotting methods inside the batch loop.

### Running both circular and non-circular routes

This is fine for checking and comparison, but for final analysis choose one route per dataset or analysis question.
