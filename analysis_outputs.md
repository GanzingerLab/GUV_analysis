# Analysis outputs

This document explains the main results stored in each `GUV` object after running the analysis.

Most calculated values are stored in:

```python
guv.analysis
```

Membrane-detection-specific values are stored in:

```python
guv.analysis.membrane
```

A `GUV` can be analysed using either the circular route or the non-circular route:

```python
guv.run_circular_analysis()
```

or:

```python
guv.run_noncircular_analysis()
```

The two routes share some early outputs, such as intensity profiles and background correction, but they produce different membrane-detection, radial-profile, angular-profile, localization, and inside-intensity outputs.

---

## Coordinate and array conventions

Images are stored as:

```text
channel x y
```

Linear intensity profiles are stored as:

```text
radial position x angle x channel
```

Angular profiles are stored as:

```text
angle x channel
```

Radial profiles are stored as:

```text
radial position x channel
```

For circular analysis, radial distances are in pixels and the x-axis is:

```python
guv.along_radius
```

For non-circular shape-normalized radial profiles, the x-axis is:

```python
guv.analysis.noncircular_normalized_along_radius
```

In the non-circular normalized axis:

```text
1 = detected membrane position
< 1 = inside the GUV
> 1 = outside the GUV
```

---

# General GUV-level outputs

## `guv.death_mark`

Type:

```text
bool
```

Meaning:

Indicates whether the GUV failed a quality-control step.

If `guv.death_mark` is `True`, the GUV is considered bad or unreliable for the current analysis route. If:

```python
settings.skip_death_marked = True
```

then later analysis steps are skipped after a GUV is death-marked.

Common reasons for death marking include:

```text
too close to the border
no_memb_peak
insufficient_membrane_signal
Low circular membrane fraction.
Low noncircular membrane fraction.
```

---

## `guv.analysis.comments`

Type:

```text
list[str]
```

Meaning:

List of warnings, quality-control comments, or reasons why a GUV was death-marked.

Examples:

```text
too close to the border
partial radial profile outside image
no_memb_peak
wide_membrane
confetti
contour_jump
irregular_contour
Low noncircular membrane fraction.
```

Some comments are fatal and mark the GUV as bad. Others are warnings and the GUV may still be usable. Which comments kill a GUV is controlled by:

```python
settings.filter.killing_comments
settings.filter.surviving_comments
```

---

# Background outputs

## `guv.analysis.background`

Shape:

```text
channel
```

Meaning:

Background intensity used for this GUV, one value per channel.

This value is used to background-correct the radial intensity profiles and to calculate background-corrected inside-GUV intensity.

Calculated by:

```python
guv.set_background()
```

The background can be local or global depending on:

```python
settings.background.method
```

Allowed methods are currently:

```text
local
global
local_then_global
```

---

## `guv.analysis.background_method`

Type:

```text
str or None
```

Meaning:

The background method actually used for this specific GUV.

Possible values after calculation are usually:

```text
local
global
```

For example, if `settings.background.method = "local_then_global"`, the package first tries local background. If local background fails, it falls back to global background and stores:

```python
guv.analysis.background_method = "global"
```

---

## `guv.analysis.intensity_profiles_corrected`

Type:

```text
bool
```

Meaning:

Indicates whether background correction has already been applied to:

```python
guv.analysis.intensity_profiles
```

This prevents the same profiles from being corrected multiple times.

Calculated/updated by:

```python
guv.correct_intensity_profiles()
```

---

# Linear intensity profile outputs

## `guv.analysis.intensity_profiles`

Shape:

```text
radial position x angle x channel
```

Meaning:

Radial intensity profiles sampled from the GUV center outwards at many angles.

These profiles are the base input for:

```text
circular membrane detection
non-circular membrane detection
radial profiles
angular profiles
membrane localization
membrane-fraction filtering
```

Calculated by:

```python
guv.calculate_intensity_profiles()
```

Then background-corrected by:

```python
guv.correct_intensity_profiles()
```

Important detail:

If a requested sampling point falls outside the image, the value is stored as:

```python
np.nan
```

This means that the value is missing, not that the intensity is zero or background.

The radial axis corresponding to the first dimension is:

```python
guv.along_radius
```

The angular axis corresponding to the second dimension is:

```python
guv.theta
```

---

# Membrane detection outputs

The membrane detection results are stored in:

```python
guv.analysis.membrane
```

The same `MembraneDetectionResult` object contains fields for both circular and non-circular analysis. Depending on which route was run, only some fields will be filled.

---

## `guv.analysis.membrane.peak_radius`

Route:

```text
circular only
```

Type:

```text
float or None
```

Meaning:

Detected circular membrane radius in pixels.

This is the radial distance from the GUV center to the detected membrane peak.

Calculated by:

```python
guv.detect_circular_membrane()
```

Used by:

```text
circular shape plotting
circular inside-GUV mask
circular intensity calculation
```

---

## `guv.analysis.membrane.peak_index`

Route:

```text
circular only
```

Type:

```text
float or int, or None
```

Meaning:

Index of the detected membrane peak in the radial profile array.

This is different from `peak_radius`:

```text
peak_index  = array position in guv.along_radius
peak_radius = physical radial distance in pixels
```

Example:

```python
radius_in_pixels = guv.along_radius[int(guv.analysis.membrane.peak_index)]
```

Used by:

```text
circular membrane-fraction filtering
circular membrane localization
```

---

## `guv.analysis.membrane.inner_border_index`

Route:

```text
circular only
```

Type:

```text
int or None
```

Meaning:

Inner radial index of the detected circular membrane band.

It is calculated from the measured width of the selected membrane peak.

Used by:

```text
circular angular profile
circular membrane localization
```

---

## `guv.analysis.membrane.outer_border_index`

Route:

```text
circular only
```

Type:

```text
int or None
```

Meaning:

Outer radial index of the detected circular membrane band.

It is calculated from the measured width of the selected membrane peak.

Used by:

```text
circular angular profile
circular membrane localization
```

---

## `guv.analysis.membrane.peak_radius_by_angle`

Route:

```text
non-circular only
```

Shape:

```text
angle
```

Meaning:

Detected membrane radius for each angle.

Each value is the distance in pixels from the GUV center to the detected non-circular membrane contour at that angle.

Calculated by:

```python
guv.detect_noncircular_membrane()
```

Used by:

```text
non-circular shape plotting
shape-normalized radial profiles
non-circular angular profiles
non-circular membrane localization
non-circular membrane-fraction filtering
```

---

## `guv.analysis.membrane.shape_x`

Route:

```text
non-circular only
```

Shape:

```text
angle
```

Meaning:

X-coordinates of the detected non-circular membrane contour in image coordinates.

Calculated from:

```python
shape_x = xc + peak_radius_by_angle * cos(theta)
```

Used by:

```text
non-circular shape plotting
non-circular inside-GUV mask
non-circular intensity calculation
```

---

## `guv.analysis.membrane.shape_y`

Route:

```text
non-circular only
```

Shape:

```text
angle
```

Meaning:

Y-coordinates of the detected non-circular membrane contour in image coordinates.

Calculated from:

```python
shape_y = yc + peak_radius_by_angle * sin(theta)
```

Used by:

```text
non-circular shape plotting
non-circular inside-GUV mask
non-circular intensity calculation
```

---

## `guv.analysis.membrane.mean_radius`

Route:

```text
non-circular mainly
```

Type:

```text
float or None
```

Meaning:

Average detected membrane radius around the non-circular contour.

Calculated as the mean of the detected membrane radii before converting them into the final contour coordinates.

Used mainly as a summary of the detected non-circular GUV size.

---

# Radial profile outputs

## `guv.analysis.circular_radial_profiles`

Route:

```text
circular only
```

Shape:

```text
radial position x channel
```

Meaning:

Angle-averaged radial intensity profile for each channel.

It is calculated by averaging:

```python
guv.analysis.intensity_profiles
```

over the angular dimension.

Calculated by:

```python
guv.calculate_circular_radial_profile()
```

X-axis:

```python
guv.along_radius
```

Current behavior:

```python
np.nanmean(intensity_profiles, axis=1)
```

So missing values from outside-image profile positions are ignored when possible.

Used by:

```text
circular membrane detection
circular profile plotting
```

---

## `guv.analysis.noncircular_radial_profiles`

Route:

```text
non-circular only
```

Shape:

```text
normalized radial position x channel
```

Meaning:

Shape-normalized radial intensity profiles.

For each angle, the radial axis is divided by the detected local membrane radius. Therefore, the membrane is aligned to:

```text
normalized distance = 1
```

This makes it possible to average radial profiles from a non-circular GUV relative to the detected local membrane contour.

Calculated by:

```python
guv.calculate_normalized_noncircular_radial_profile()
```

X-axis:

```python
guv.analysis.noncircular_normalized_along_radius
```

Used by:

```text
non-circular profile plotting
visual inspection of inside/membrane/outside signal relative to the detected shape
```

---

## `guv.analysis.noncircular_normalized_along_radius`

Route:

```text
non-circular only
```

Shape:

```text
normalized radial position
```

Meaning:

Common normalized radial axis for:

```python
guv.analysis.noncircular_radial_profiles
```

Interpretation:

```text
0   = GUV center
1   = detected membrane position
> 1 = outside the detected membrane
```

---

# Angular profile outputs

## `guv.analysis.circular_uncorrected_angular_profiles`

Route:

```text
circular only
```

Shape:

```text
angle x channel
```

Meaning:

Angular membrane intensity profiles before angular flattening/correction.

For every angle, the signal is averaged over the circular membrane band between:

```python
guv.analysis.membrane.inner_border_index
```

and:

```python
guv.analysis.membrane.outer_border_index
```

Calculated by:

```python
guv.calculate_circular_angular_profile()
```

---

## `guv.analysis.circular_angular_profiles`

Route:

```text
circular only
```

Shape:

```text
angle x channel
```

Meaning:

Angular membrane intensity profiles after correction of the membrane channel, if angular profile flattening is enabled.

If:

```python
settings.profiles.flatten_angular_profiles = True
```

then the GUV membrane channel is corrected for broad angular intensity modulation in the poles of the GUVs in the y axis.

Only the GUV membrane channel is replaced by the flattened profile. Other channels remain as their original membrane angular profiles.

---

## `guv.analysis.noncircular_uncorrected_angular_profiles`

Route:

```text
non-circular only
```

Shape:

```text
angle x channel
```

Meaning:

Angular membrane intensity profiles before angular flattening/correction, calculated along the detected non-circular membrane contour.

For every angle, the signal is averaged over a small radial window centered on:

```python
guv.analysis.membrane.peak_radius_by_angle
```

The width of this window is controlled by:

```python
settings.profiles.noncircular_membrane_width_pixels
```

Calculated by:

```python
guv.calculate_noncircular_angular_profile()
```

---

## `guv.analysis.noncircular_angular_profiles`

Route:

```text
non-circular only
```

Shape:

```text
angle x channel
```

Meaning:

Angular membrane intensity profiles after correction of the membrane channel, if angular profile flattening is enabled.

If:

```python
settings.profiles.flatten_angular_profiles = True
```

then the GUV membrane channel is corrected for broad angular intensity modulation.

Only the GUV membrane channel is replaced by the flattened profile. Other channels remain as their original membrane angular profiles.

---

# Membrane-fraction outputs

## `guv.analysis.circular_fraction_membrane`

Route:

```text
circular only
```

Type:

```text
float or None
```

Meaning:

Fraction of angles where the detected circular membrane has enough local membrane support.

Calculated by:

```python
guv.filter_circular_fraction()
```

How it is calculated:

For each angle, the function compares the membrane-channel signal near the detected membrane with nearby baseline regions inside and outside the membrane.

The per-angle support score is:

```text
support = membrane_prominence / local_noise
```

where:

```text
membrane_prominence = membrane signal - local baseline
```

A given angle passes when:

```python
support >= settings.profiles.min_radial_membrane_support
```

The final value is the fraction of valid angles that pass.

The GUV is death-marked when:

```python
fraction_membrane < settings.profiles.min_fraction_membrane
```

Low values can indicate:

```text
weak membrane signal
bad membrane detection
membrane only visible over part of the contour
insufficient valid radial profile for the inner/membrane/outer windows
```

---

## `guv.analysis.noncircular_fraction_membrane`

Route:

```text
non-circular only
```

Type:

```text
float or None
```

Meaning:

Fraction of angles where the detected non-circular membrane has enough local membrane support.

Calculated by:

```python
guv.filter_noncircular_fraction()
```

How it is calculated:

The detected membrane radius for every angle is first converted to the closest radial profile index. Then the same membrane-support calculation as above is applied angle by angle.

Each angle needs three windows:

```text
inner baseline window
membrane window
outer baseline window
```

If one of these windows does not fit inside the available radial profile, or contains no finite values, that angle is skipped.

The final value is the fraction of valid angles whose support is high enough.

The GUV is death-marked when:

```python
fraction_membrane < settings.profiles.min_fraction_membrane
```

Low values can indicate:

```text
weak membrane signal
bad non-circular contour
partial membrane signal
GUV close to the image edge
not enough radial profile available outside the membrane
```

---

# Localization outputs

## `guv.analysis.circular_memb_localization`

Route:

```text
circular only
```

Type:

```text
dict
```

Meaning:

Membrane-localization values for all non-GUV channels, calculated using the circular membrane detection.

The GUV membrane marker channel is excluded. The returned dictionary maps channel index to localization value, for example:

```python
{0: 0.45, 2: 0.12}
```

if `settings.guv_ch = 1`.

Calculated by:

```python
guv.calculate_circular_membrane_localization()
```

How it is calculated:

For each non-GUV channel, the package calculates:

```text
membrane_signal = typical signal in the detected circular membrane band
centre_signal   = average signal in the central region
```

Then:

```text
localization = (membrane_signal - centre_signal) / membrane_signal
```

Interpretation:

```text
close to 1  = signal is much higher at the membrane than in the center
close to 0  = membrane and center are similar
< 0         = center signal is higher than membrane signal
```

---

## `guv.analysis.noncircular_memb_localization`

Route:

```text
non-circular only
```

Type:

```text
dict
```

Meaning:

Membrane-localization values for all non-GUV channels, calculated using the detected non-circular membrane contour.

The GUV membrane marker channel is excluded. The returned dictionary maps channel index to localization value.

Calculated by:

```python
guv.calculate_noncircular_membrane_localization()
```

How it is calculated:

For each angle, the package finds the radial sample closest to the detected non-circular membrane radius and averages a small window around it. It also calculates the center signal for that angle.

Across angles, the membrane and center signals are combined, and the localization value is calculated as:

```text
localization = (membrane_signal - centre_signal) / membrane_signal
```

Interpretation:

```text
close to 1  = signal is much higher at the membrane than in the center
close to 0  = membrane and center are similar
< 0         = center signal is higher than membrane signal
```

---

# Inside-intensity outputs

## `guv.analysis.circular_inside_intensity`

Route:

```text
circular only
```

Shape:

```text
channel
```

Meaning:

Background-corrected mean intensity inside the circular GUV mask, one value per channel.

Calculated by:

```python
guv.calculate_circular_intensity()
```

How it is calculated:

1. A circular mask is created using:

```python
guv.xc
guv.yc
guv.analysis.membrane.peak_radius
```

2. The raw mean intensity inside the mask is calculated for every channel.

3. The GUV background is subtracted:

```text
corrected_inside_intensity = raw_inside_intensity - background
```

Only the corrected intensity is currently stored in:

```python
guv.analysis.circular_inside_intensity
```

---

## `guv.analysis.noncircular_inside_intensity`

Route:

```text
non-circular only
```

Shape:

```text
channel
```

Meaning:

Background-corrected mean intensity inside the non-circular GUV mask, one value per channel.

Calculated by:

```python
guv.calculate_noncircular_intensity()
```

How it is calculated:

1. A non-circular mask is created from the detected contour:

```python
guv.analysis.membrane.shape_x
guv.analysis.membrane.shape_y
```

2. The raw mean intensity inside the mask is calculated for every channel.

3. The GUV background is subtracted:

```text
corrected_inside_intensity = raw_inside_intensity - background
```

Only the corrected intensity is currently stored in:

```python
guv.analysis.noncircular_inside_intensity
```

---

# What is stored for each analysis route?

## Circular route

Running:

```python
guv.run_circular_analysis()
```

usually fills:

```python
guv.analysis.background
guv.analysis.background_method
guv.analysis.intensity_profiles_corrected
guv.analysis.intensity_profiles

guv.analysis.circular_radial_profiles

guv.analysis.membrane.peak_radius
guv.analysis.membrane.peak_index
guv.analysis.membrane.inner_border_index
guv.analysis.membrane.outer_border_index

guv.analysis.circular_fraction_membrane
guv.analysis.circular_uncorrected_angular_profiles
guv.analysis.circular_angular_profiles
guv.analysis.circular_memb_localization
guv.analysis.circular_inside_intensity

guv.analysis.comments
guv.death_mark
```

## Non-circular route

Running:

```python
guv.run_noncircular_analysis()
```

usually fills:

```python
guv.analysis.background
guv.analysis.background_method
guv.analysis.intensity_profiles_corrected
guv.analysis.intensity_profiles

guv.analysis.membrane.peak_radius_by_angle
guv.analysis.membrane.shape_x
guv.analysis.membrane.shape_y
guv.analysis.membrane.mean_radius

guv.analysis.noncircular_fraction_membrane
guv.analysis.noncircular_radial_profiles
guv.analysis.noncircular_normalized_along_radius
guv.analysis.noncircular_uncorrected_angular_profiles
guv.analysis.noncircular_angular_profiles
guv.analysis.noncircular_memb_localization
guv.analysis.noncircular_inside_intensity

guv.analysis.comments
guv.death_mark
```

---

# Useful examples

## Inspect one analysed GUV

```python
guv = image_analysis.guvs[3]

print("Death mark:", guv.death_mark)
print("Comments:", guv.analysis.comments)
print("Background:", guv.analysis.background)
print("Inside intensity:", guv.analysis.noncircular_inside_intensity)
```

## Extract one output from all good GUVs

```python
inside_intensity = image_analysis.extract_parameter(
    "analysis.noncircular_inside_intensity",
    GUVs="good",
)
```

## Plot inside intensity from two channels

```python
int_ch0 = [val[0] for val in inside_intensity.values() if val is not None]
int_ch1 = [val[1] for val in inside_intensity.values() if val is not None]

plt.scatter(int_ch0, int_ch1)
plt.xlabel("Inside intensity channel 0")
plt.ylabel("Inside intensity channel 1")
plt.show()
```

## Check whether partial radial profiles contain missing values

```python
profiles = guv.analysis.intensity_profiles
ch = guv.settings.guv_ch

print("Has NaNs:", np.isnan(profiles[:, :, ch]).any())
print("Fraction NaN:", np.isnan(profiles[:, :, ch]).mean())

cmap = plt.cm.gray.copy()
cmap.set_bad(color="red")

plt.imshow(
    np.ma.masked_invalid(profiles[:, :, ch]),
    aspect="auto",
    origin="lower",
    cmap=cmap,
)
plt.xlabel("Angle index")
plt.ylabel("Radial position index")
plt.title(f"GUV {guv.id} intensity profile - NaNs in red")
plt.colorbar(label="Intensity")
plt.show()
```

---

# Notes and possible future improvements

- The same `guv.analysis.membrane` object is used for both circular and non-circular membrane detection. For final analysis, it is best to run only one.
- `guv.analysis.intensity_profiles` can contain `NaN` values when part of the requested radial profile falls outside the image. These values represent missing data, not zero intensity and not background.
- `circular_inside_intensity` and `noncircular_inside_intensity` currently store the background-corrected mean intensity only. The raw mean intensity is calculated internally but not stored in `GUVAnalysis`.
- Some comments indicate failed analysis, while others are warnings. The behavior can be controlled through `settings.filter`.
- Low membrane fraction can mean weak membrane signal, but it can also mean that too few valid radial windows were available, especially for GUVs close to image borders.
