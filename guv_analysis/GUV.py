import os
import sys
import csv
from glob import glob
from dataclasses import dataclass, field
from functools import wraps
from collections.abc import Callable


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bioio import BioImage
from scipy.signal import find_peaks, peak_widths
import tifffile as tif
from tqdm import tqdm

from guv_analysis.background import background_correction, dilate_vesicle_mask, local_background
from guv_analysis.io_tools import get_output_folder
from guv_analysis.membrane_detection import detect_circular_GUV, detect_noncircular_GUV
from guv_analysis.plotting import plot_profile_and_zoom, save_separate_profile_plots, plot_detected_guv_shape, plot_shape_normalized_profiles_crops_and_angles
from guv_analysis.profiles import angular_profile, linear_profiles, radial_profile, trim_central_profiles, choose_num_angles, normalized_radial_profile_from_detected_shape, angular_profile_from_detected_shape, radial_positions_to_indices
from guv_analysis.signal_quantification import circular_membrane_localization, noncircular_membrane_localization, calculate_inside_intensity
from guv_analysis.settings import AnalysisSettings
from guv_analysis.image_view import GUVImageView
from guv_analysis.masks import circular_GUV_mask, noncircular_GUV_mask
from guv_analysis.filter import membrane_fraction

def skip_if_dead(method):
    """
    Decroator to skip a GUV analysis method when the vesicle is already death-marked.

    The method is skipped only when both conditions are true:

    - ``self.death_mark`` is ``True``.
    - ``self.settings.skip_death_marked`` is ``True``.

    Setting ``skip_death_marked`` to ``False`` allows analysis methods to run
    even when the GUV has already been death-marked. The death mark itself is
    not cleared or modified.
    """
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        should_skip = (self.death_mark and self.settings.skip_death_marked)

        if should_skip:
            print(f"Skipping dead GUV {self.id}: {method.__name__}")
            return None

        return method(self, *args, **kwargs)

    return wrapper


@dataclass
class MembraneDetectionResult:
    # --- Circular membrane detection results ---
    peak_radius: float | None = None # Refined radial position obtained by quadratic interpolation. In px.
    peak_index: float | None = None # Integer position of the detected peak in the radial intensity profile. Indicates peak location within along radius. 
    inner_border_index: int | None = None #same for the inner border
    outer_border_index: int | None = None #same for the outer border
     # --- Non-circular membrane detection results ---
    peak_radius_by_angle: np.ndarray | None = None #Radius of the membrane at each angle
    shape_x: np.ndarray | None = None #array contianing the cartesian coordinates of the full membrane contour on the X diraction
    shape_y: np.ndarray | None = None #array contianing the cartesian coordinates of the full membrane contour on the Y diraction
    mean_radius: float | None = None #Average of the detected radii around the contour
    

@dataclass
class GUVAnalysis:
    background: np.ndarray | None = None  # Background intensity used for this GUV, one value per channel
    background_method: str | None = None  # Background method actually used; usually "local" or "global"
    intensity_profiles_corrected: bool = False  # True after background correction has been applied

    profile_extended_for_padding: bool = False
    effective_length_excess: float | None = None

    intensity_profiles: np.ndarray | None = None  # Radial intensity profiles; shape radial_positions x angles x channels

    circular_radial_profiles: np.ndarray | None = None   # Angle-averaged radial profiles for the circular route; x-axis is along_radius in pixels. Array containing one per channel. 
    noncircular_radial_profiles: np.ndarray | None = None  # Shape-normalized radial profiles for the non-circular route; x-axis is noncircular_normalized_along_radius
    noncircular_normalized_along_radius: np.ndarray | None = None  # Normalized radial axis for non-circular profiles; it is like along_radius but the membrane is aligned at 1.  

    circular_angular_profiles: np.ndarray | None = None  # Circular membrane angular profiles; shape angles x channels
    circular_uncorrected_angular_profiles: np.ndarray | None = None  # Circular angular profiles before angular flattening/correction
    noncircular_angular_profiles: np.ndarray | None = None  # Non-circular membrane angular profiles; shape angles x channels
    noncircular_uncorrected_angular_profiles: np.ndarray | None = None  # Non-circular angular profiles before angular flattening/correction

    membrane: MembraneDetectionResult = field(default_factory=MembraneDetectionResult)  # Membrane detection results shared by circular/non-circular routes

    circular_fraction_membrane: float | None = None  # Fraction of angles passing the circular membrane-support filter
    noncircular_fraction_membrane: float | None = None  # Fraction of angles passing the non-circular membrane-support filter
    
    circular_memb_localization: dict | None = None  # Circular membrane localization values, usually one value per non-GUV channel
    noncircular_memb_localization: dict | None = None  # Non-circular membrane localization values, usually one value per non-GUV channel

    circular_inside_intensity: np.ndarray | None = None  # Inside-GUV intensity from the circular route, one value per channel
    noncircular_inside_intensity: np.ndarray | None = None  # Inside-GUV intensity from the non-circular route, one value per channel

    comments: list[str] = field(default_factory=list)  # Warnings and quality-control comments for this GUV

@dataclass(kw_only=True)
class GUV:
    """
    Analysis object for one detected GUV.

    A GUV object stores the vesicle position, image reference, settings,
    intermediate analysis results, final measurements, and comments.

    Most analysis methods modify ``self.analysis`` in place. For example,
    after running ``run_noncircular_analysis()``, results such as the detected
    membrane contour, radial profiles, angular profiles, localization values,
    and inside intensity are stored in ``self.analysis``.

    Main workflow
    -------------
    Circular analysis:
        guv.run_circular_analysis()

    Non-circular analysis:
        guv.run_noncircular_analysis()

    Important attributes
    --------------------
    id : int
        Vesicle ID from the detection CSV.

    xc, yc : float
        Vesicle center coordinates in pixels.

    radius : float
        Approximate vesicle radius in pixels from the detection CSV.

    death_mark : bool
        True if the GUV failed a quality-control step.

    analysis : GUVAnalysis
        Object where all calculated results are stored.
    """
    id: int
    xc: float
    yc: float
    radius: float

    settings: AnalysisSettings
    num_angles: int = field(init=False)

    image_view: GUVImageView = field(repr=False)

    analysis: GUVAnalysis = field(default_factory=GUVAnalysis)
    death_mark: bool = False
    ves_coordinates: np.ndarray = field(init=False, repr=False)
    on_death_marked: Callable[["GUV"], None] | None = field(default=None, repr=False)


    def __post_init__(self) -> None:
        self.ves_coordinates = np.array([self.id, self.xc, self.yc, self.radius,], dtype=float)
        self.num_angles = choose_num_angles(self.radius, self.settings.profiles.target_arc_spacing)

    @property
    def full_along_radius(self) -> np.ndarray:
        """
        Return the complete radial sampling axis.

        The profile uses the length requested through ``length_excess`` unless
        that length provides insufficient samples outside the approximate GUV
        radius. For small GUVs, the profile is automatically extended to include
        ``minimum_outer_padding_pixels`` beyond the approximate membrane.
        """
        settings = self.settings.profiles
        step = settings.profile_step

        # Calculate the last sample index produced by the user-requested length excess.
        requested_last = int(np.floor(settings.length_excess * self.radius / step))

        # Ensure enough samples remain outside small GUVs for the membrane window,
        # baseline gap, and outer-baseline window.
        minimum_last = (int(np.floor(self.radius / step)) + int(np.ceil(settings.minimum_outer_padding_pixels / step)))

        # Use the requested length unless it provides insufficient outward padding.
        last_sample = max(requested_last, minimum_last)

        # Record the actual profile length used for this GUV.
        self.analysis.profile_extended_for_padding = minimum_last > requested_last
        self.analysis.effective_length_excess = (last_sample * step) / self.radius

        # Convert the radial sample indices into radial positions.
        return np.arange(last_sample + 1) * step

    @property
    def samples_to_remove(self) -> int:
        return int(np.ceil(self.settings.profiles.pixels_to_remove/ self.settings.profiles.profile_step)
        )

    @property
    def along_radius(self) -> np.ndarray:
        return self.full_along_radius[self.samples_to_remove:]

    @property
    def theta(self):
        return np.linspace(0, 2 * np.pi, self.num_angles, endpoint=False)

    def plot_crop(self, channel=None, length_excess=None, save_path=None, show=True) -> None:
        """Show a cropped image around this GUV."""

        if channel is None:
            channel = self.settings.guv_ch

        if length_excess is None:
            length_excess = self.settings.profiles.length_excess

        view_radius = length_excess * self.radius

        x_min = max(0, int(self.xc - view_radius))
        x_max = min(self.image_view.image.shape[2], int(self.xc + view_radius + 1))
        y_min = max(0, int(self.yc - view_radius))
        y_max = min(self.image_view.image.shape[1], int(self.yc + view_radius + 1))

        fig, ax = plt.subplots()
        ax.imshow(self.image_view.image[channel, y_min:y_max, x_min:x_max], cmap="gray")
        ax.set_title(f"GUV {self.id} crop - channel {channel}")
        ax.axis("equal")
        ax.axis("off")

        if save_path is not None:
            fig.savefig(save_path, dpi=self.settings.plotting.dpi, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close(fig)

    @skip_if_dead
    def calculate_intensity_profiles(self) -> None: 
        intensity_profiles, death_mark, comment = linear_profiles(
            self.image_view.image, 
            self.ves_coordinates, 
            self.full_along_radius,
            self.theta, self.settings.profiles)
        if death_mark:
            self.mark_dead()
        self.analysis.comments.extend(comment)
        self.analysis.intensity_profiles = trim_central_profiles(intensity_profiles, self.samples_to_remove)

    def calculate_local_background(self) -> None:
        dilated_mask = dilate_vesicle_mask(self.image_view.global_mask, iterations=3)
        return local_background(self.image_view.image, self.ves_coordinates, dilated_mask, 
            self.settings.background.inner_margin, self.settings.background.outer_margin)
    @skip_if_dead
    def set_background(self) -> None:
        method = self.settings.background.method
        valid_methods = ("local", "global", "local_then_global")

        if method not in valid_methods:
            raise ValueError(f"Invalid method {method!r}. Choose one of: {', '.join(valid_methods)}")

        if (method in ("global", "local_then_global") and self.image_view.global_background is None):
            raise RuntimeError(f"Background method {method!r} requires a global background, but it has not been calculated.")

        if method == "local":
            try:
                self.analysis.background = self.calculate_local_background()
                self.analysis.background_method = "local"
            except ValueError as error:
                self.analysis.comments.append("local bckg failed")
                self.mark_dead()

        elif method == "global":
            self.analysis.background = self.image_view.global_background
            self.analysis.background_method = "global"

        else:  # local_then_global
            try:
                self.analysis.background = self.calculate_local_background()
                self.analysis.background_method = "local"
            except ValueError as error:
                self.analysis.comments.append(f"{error}-Using global background.")
                self.analysis.background = self.image_view.global_background
                self.analysis.background_method = "global"
    @skip_if_dead
    def correct_intensity_profiles(self) -> None:
        if self.analysis.intensity_profiles is None:
            raise RuntimeError("Intensity profiles have not been calculated.")

        if self.analysis.background is None:
            raise RuntimeError("Background has not been set.")
        self.analysis.intensity_profiles = background_correction(self.analysis.intensity_profiles, self.analysis.background)
        self.analysis.intensity_profiles_corrected = True

    @skip_if_dead
    def calculate_circular_radial_profile(self) -> None:
        self.analysis.circular_radial_profiles = radial_profile(self.analysis.intensity_profiles)
    #TODO: normalize circular profiles
    
    @skip_if_dead
    def detect_circular_membrane(self) -> None:
        if self.analysis.circular_radial_profiles is None:
            raise RuntimeError("Circular radial profile has not been calculated.")

        detection= self.analysis.membrane
        detection.peak_radius, detection.peak_index, detection.inner_border_index, detection.outer_border_index, comments, detection_failed = detect_circular_GUV(
            self.analysis.circular_radial_profiles[:,self.settings.guv_ch], self.along_radius, self.radius, self.settings.circular_membrane)
        if detection_failed:
            self.mark_dead()
        self.analysis.comments.extend(comments)
        self.analysis.membrane = detection

    @skip_if_dead
    def detect_noncircular_membrane(self) -> None:
        if self.analysis.intensity_profiles is None:
                    raise RuntimeError("Intensity profiles have not been calculated.")
        detection= self.analysis.membrane
        detection.shape_x, detection.shape_y, detection.peak_radius_by_angle, detection.mean_radius, comments, detection_failed = detect_noncircular_GUV(
            self.analysis.intensity_profiles, self.along_radius, self.theta, self.ves_coordinates, self.settings.noncircular_membrane, self.settings.guv_ch)
        if detection_failed:
            self.mark_dead()
        self.analysis.comments.extend(comments)
        self.analysis.membrane = detection

    @skip_if_dead
    def filter_circular_fraction(self, baseline_gap: int = 2) -> None:
        if self.analysis.membrane.peak_index is None:
            raise RuntimeError("Circular membrane not calculated.")

        support, prominence, fraction_membrane, death_mark = membrane_fraction(
            intensity_profiles=self.analysis.intensity_profiles,
            peak_index_by_angle=self.analysis.membrane.peak_index,
            global_background=self.image_view.global_background[self.settings.guv_ch],
            profile_settings=self.settings.profiles,
            guv_channel=self.settings.guv_ch,
            baseline_gap=baseline_gap,
        )
        self.analysis.circular_fraction_membrane = fraction_membrane
        if death_mark:
            self.mark_dead()
            self.analysis.comments.append("Low circular membrane fraction.")

    @skip_if_dead
    def filter_noncircular_fraction(self, baseline_gap:int = 3) -> None:
        if self.analysis.membrane.peak_radius_by_angle is None: 
            raise RuntimeError("Non-circular membrane not calculated.")  

        peak_index_by_angle = radial_positions_to_indices(self.analysis.membrane.peak_radius_by_angle, self.along_radius)

        support, prominence, fraction_membrane, death_mark = membrane_fraction(
            self.analysis.intensity_profiles, 
            peak_index_by_angle, 
            self.image_view.global_background[self.settings.guv_ch], 
            self.settings.profiles, 
            self.settings.guv_ch, baseline_gap=baseline_gap
        )
        self.analysis.noncircular_fraction_membrane = fraction_membrane
        if death_mark:
            self.mark_dead()
            self.analysis.comments.append("Low noncircular membrane fraction.")
        
    @skip_if_dead
    def calculate_normalized_noncircular_radial_profile(self, number_radial_points = 100) -> None:
        if self.analysis.intensity_profiles is None:
            raise RuntimeError("Intensity profiles have not been calculated.")
        if self.analysis.membrane.peak_radius_by_angle is None: 
            raise RuntimeError("Non-circular membrane not calculated.")        
        
        self.analysis.noncircular_radial_profiles, self.analysis.noncircular_normalized_along_radius, _ , comments = normalized_radial_profile_from_detected_shape(
            self.analysis.intensity_profiles, self.along_radius, self.analysis.membrane.peak_radius_by_angle, self.settings.guv_ch, self.settings.circular_membrane, self.settings.profiles.length_excess, number_radial_points)
        self.analysis.comments.extend(comments)

    @skip_if_dead
    def calculate_circular_angular_profile(self) -> None: 
        if self.analysis.intensity_profiles is None:
            raise RuntimeError("Intensity profiles have not been calculated.")
        if self.analysis.membrane.inner_border_index is None: 
            raise RuntimeError("Circular membrane not calculated.")

        angular_profiles, flattened_profile, death_mark, comments  = angular_profile(
            self.analysis.intensity_profiles, self.analysis.membrane.inner_border_index, self.analysis.membrane.outer_border_index, self.settings.profiles, self.settings.guv_ch)
        self.analysis.circular_uncorrected_angular_profiles = angular_profiles
        membrane_corrected_angular_profiles = angular_profiles.copy()
        membrane_corrected_angular_profiles[:, self.settings.guv_ch] = flattened_profile
        self.analysis.circular_angular_profiles = membrane_corrected_angular_profiles
        self.analysis.comments.extend(comments)
        if death_mark:
            self.mark_dead()

    @skip_if_dead
    def calculate_noncircular_angular_profile(self) -> None:
        if self.analysis.intensity_profiles is None:
            raise RuntimeError("Intensity profiles have not been calculated.")
        if self.analysis.membrane.peak_radius_by_angle is None: 
            raise RuntimeError("Non-circular membrane not calculated.")

        angular_profiles, flattened_profile, death_mark, comments = angular_profile_from_detected_shape(
            self.analysis.intensity_profiles, self.along_radius, self.analysis.membrane.peak_radius_by_angle, self.settings.profiles, self.settings.guv_ch)
        self.analysis.noncircular_uncorrected_angular_profiles = angular_profiles
        membrane_corrected_angular_profiles = angular_profiles.copy()
        membrane_corrected_angular_profiles[:, self.settings.guv_ch] = flattened_profile
        self.analysis.noncircular_angular_profiles = membrane_corrected_angular_profiles
        self.analysis.comments.extend(comments)
        if death_mark:
            self.mark_dead()

    @skip_if_dead
    def calculate_circular_membrane_localization(self) -> None:
        if self.analysis.intensity_profiles is None:
            raise RuntimeError("Intensity profiles have not been calculated.")
        if self.analysis.membrane.peak_index is None: 
            raise RuntimeError("Circular membrane not calculated.")
        
        self.analysis.circular_memb_localization, comment = circular_membrane_localization(
            self.analysis.intensity_profiles, self.analysis.membrane.peak_index, self.analysis.membrane.inner_border_index, 
            self.analysis.membrane.outer_border_index, self.settings.localization.centre_size, self.settings.guv_ch)
        self.analysis.comments.extend(comment)

    @skip_if_dead
    def calculate_noncircular_membrane_localization(self) -> None:
        if self.analysis.intensity_profiles is None:
            raise RuntimeError("Intensity profiles have not been calculated.")
        if self.analysis.membrane.peak_radius_by_angle is None: 
            raise RuntimeError("Noncircular membrane not calculated.")
        
        self.analysis.noncircular_memb_localization, comment = noncircular_membrane_localization(
            self.analysis.intensity_profiles, self.along_radius, self.analysis.membrane.peak_radius_by_angle, 
            self.settings.profiles.noncircular_membrane_width_pixels, self.settings.localization.centre_size, self.settings.guv_ch)
        self.analysis.comments.extend(comment)

    @skip_if_dead
    def calculate_circular_intensity(self) -> None:
        """Calculate mean intensity inside the detected circular GUV."""
        detection = self.analysis.membrane
        radius = detection.peak_radius

        if radius is None or not np.isfinite(radius) or radius <= 0:
            self.analysis.comments.append("invalid_circular_radius")
            self.mark_dead()
            return

        mask = circular_GUV_mask(self.image_view.image.shape[1:], self.xc, self.yc, radius)

        if not np.any(mask):
            self.analysis.comments.append("empty_circular_mask")
            self.mark_dead()
            return

        _, self.analysis.circular_inside_intensity = calculate_inside_intensity(self.image_view.image, mask, self.analysis.background)

    @skip_if_dead
    def calculate_noncircular_intensity(self) -> None:
        detection = self.analysis.membrane
        if detection.shape_x is None:
            raise RuntimeError("Noncircular membrane has not been detected.")

        mask = noncircular_GUV_mask(self.image_view.image.shape[1:], detection.shape_x, detection.shape_y)
        _, self.analysis.noncircular_inside_intensity = calculate_inside_intensity(self.image_view.image, mask, self.analysis.background)

    def plot_circular_profiles(self, size_view=1.5, save_path=None, show=True) -> None:
        """Plot the image crops and calculated circular radial and angular profiles."""
        if self.analysis.circular_radial_profiles is None:
            raise RuntimeError("Circular radial profiles have not been calculated.")
        channels = range(self.image_view.image.shape[0])
        plot_profile_and_zoom(self.image_view.image, self.ves_coordinates, self.analysis.circular_radial_profiles, self.along_radius, angular_profiles=self.analysis.circular_angular_profiles, size_view=size_view, channels=channels, save_path=save_path, show=show)

    def plot_noncircular_profiles(self, size_view=1.5, save_path=None, show=True) -> None:
        """Plot image crops and shape-normalized radial and angular profiles."""
        if self.analysis.noncircular_radial_profiles is None or self.analysis.noncircular_normalized_along_radius is None:
            print("Noncircular radial profiles have not been calculated.")
            return
        channels = range(self.image_view.image.shape[0])
        plot_shape_normalized_profiles_crops_and_angles(self.image_view.image, self.ves_coordinates, self.analysis.noncircular_radial_profiles, self.analysis.noncircular_normalized_along_radius, self.analysis.noncircular_angular_profiles, channels=channels, size_view=size_view, title=f"GUV {self.id} shape-normalized profiles", save_path=save_path, show=show)

    def plot_circular_shape(self, size_view=1.5, save_path=None, show=True) -> None:
        """Show the detected circular membrane over the membrane channel."""
        detection = self.analysis.membrane
        if detection.peak_radius is None:
            raise RuntimeError("Circular membrane has not been detected.")

        shape_x = self.xc + detection.peak_radius * np.cos(self.theta)
        shape_y = self.yc + detection.peak_radius * np.sin(self.theta)
        plot_detected_guv_shape(self.image_view.image, self.settings.guv_ch, self.ves_coordinates, shape_x, shape_y, size_view=size_view, title=f"GUV {self.id} circular membrane", save_path=save_path, show=show)

    def plot_noncircular_shape(self, size_view=1.5, save_path=None, show=True) -> None:
        """Plot the detected noncircular membrane contour over the membrane channel."""
        detection = self.analysis.membrane

        if detection.shape_x is None or detection.shape_y is None:
            raise RuntimeError("Noncircular membrane has not been detected.")

        plot_detected_guv_shape(self.image_view.image, self.settings.guv_ch, self.ves_coordinates, detection.shape_x, detection.shape_y, size_view=size_view, title=f"GUV {self.id} detected shape", save_path=save_path, show=show)

    @skip_if_dead
    def run_circular_analysis(self) -> None:
        """
        Run the full circular GUV analysis workflow.

        The workflow calculates radial intensity profiles, estimates background,
        background-corrects the profiles, detects the circular membrane position,
        filters weak membrane detections, calculates angular profiles, calculates
        membrane localization, and calculates inside intensity.

        Results are stored in ``self.analysis``.

        If one step fails a quality-control check, the GUV is death-marked and
        later steps may be skipped depending on ``settings.skip_death_marked``.
        """
        self._run_steps(
            self._prepare_intensity_profiles,
            self.calculate_circular_radial_profile,
            self.detect_circular_membrane,
            self.filter_circular_fraction,
            self.calculate_circular_angular_profile,
            self.calculate_circular_membrane_localization,
            self.calculate_circular_intensity,
        )

    @skip_if_dead
    def run_noncircular_analysis(self) -> None:
        """
        Run the full non-circular GUV analysis workflow.

        The workflow calculates radial intensity profiles, estimates background,
        background-corrects the profiles, detects a non-circular membrane contour,
        filters weak membrane detections, calculates shape-normalized radial
        profiles, calculates angular membrane profiles, calculates membrane
        localization, and calculates inside intensity.

        Results are stored in ``self.analysis``.

        The detected non-circular membrane is stored as:
            self.analysis.membrane.shape_x
            self.analysis.membrane.shape_y
            self.analysis.membrane.peak_radius_by_angle
        """
        self._run_steps(
            self._prepare_intensity_profiles,
            self.detect_noncircular_membrane,
            self.filter_noncircular_fraction,
            self.calculate_normalized_noncircular_radial_profile,
            self.calculate_noncircular_angular_profile,
            self.calculate_noncircular_membrane_localization,
            self.calculate_noncircular_intensity,
        )

    @skip_if_dead
    def save_circular_profile_plots(self, output_folder, channels=None) -> None:
        """
        Save separate circular radial and angular profile plots for this GUV.
        """

        if self.analysis.circular_radial_profiles is None:
            raise RuntimeError("Circular radial profiles not calculated.")

        if self.analysis.circular_angular_profiles is None:
            raise RuntimeError("Circular angular profiles not calculated.")

        if channels is None:
            channels = range(self.image_view.num_channels)

        save_separate_profile_plots(
            radial_profiles=self.analysis.circular_radial_profiles,
            along_radius=self.along_radius,
            angular_profiles=self.analysis.circular_angular_profiles,
            vesicle_id=self.id,
            channels=channels,
            output_folder=output_folder,
            file_prefix="circular_"
        )

    @skip_if_dead
    def save_noncircular_profile_plots(self, output_folder, channels=None) -> None:
        """
        Save separate non-circular radial and angular profile plots for this GUV.
        """

        if self.analysis.noncircular_radial_profiles is None:
            raise RuntimeError("Non-circular radial profiles not calculated.")

        if self.analysis.noncircular_angular_profiles is None:
            raise RuntimeError("Non-circular angular profiles not calculated.")

        if channels is None:
            channels = range(self.image_view.num_channels)

        save_separate_profile_plots(
            radial_profiles=self.analysis.noncircular_radial_profiles,
            along_radius=self.along_radius,
            angular_profiles=self.analysis.noncircular_angular_profiles,
            vesicle_id=self.id,
            channels=channels,
            output_folder=output_folder,
            file_prefix="noncircular_"
        )

    def _prepare_intensity_profiles(self) -> None:
        """Calculate and background-correct the linear intensity profiles when needed."""
        if self.analysis.intensity_profiles is None:
            self.calculate_intensity_profiles()

        if self.analysis.background is None:
            self.set_background()

        if not self.analysis.intensity_profiles_corrected:
            self.correct_intensity_profiles()

    def _run_steps(self, *steps) -> None:
        """
        Run analysis steps in order and stop when the GUV becomes death-marked.

        When execution stops, print the name of the step that set or preserved the
        death mark.
        """
        for step in steps:
            step()

            if self.death_mark and self.settings.skip_death_marked:
                print(
                    f"Stopping analysis for dead GUV {self.id} "
                    f"after step: {step.__name__}"
                )
                return
    def mark_dead(self) -> None:
        """
        Mark the GUV as dead and notify its owner on the first transition.

        Repeated calls have no effect.
        """
        if self.death_mark:
            return

        self.death_mark = True

        if self.on_death_marked is not None:
            self.on_death_marked(self)
    
