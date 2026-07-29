import os
import sys
import csv
from glob import glob
from dataclasses import dataclass, field

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
from guv_analysis.plotting import plot_profile_and_zoom, save_separate_profile_plots
from guv_analysis.profiles import angular_profile, linear_profiles, radial_profile, trim_central_profiles, choose_num_angles, normalized_radial_profile_from_detected_shape, angular_profile_from_detected_shape
from guv_analysis.signal_quantification import circular_membrane_localization, noncircular_membrane_localization, calculate_inside_intensity
from guv_analysis.settings import AnalysisSettings
from guv_analysis.image_view import GUVImageView
from guv_analysis.masks import circular_GUV_mask, noncircular_GUV_mask



@dataclass
class MembraneDetectionResult:
    comments: list[str] = field(default_factory=list)

    peak_radius: float | None = None
    peak_index: float | None = None
    inner_border_index: int | None = None
    outer_border_index: int | None = None

    peak_radius_by_angle: np.ndarray | None = None
    shape_x: np.ndarray | None = None
    shape_y: np.ndarray | None = None
    mean_radius: float | None = None
    

@dataclass
class GUVAnalysis:
    background: np.ndarray | None = None
    background_method: str | None = None

    intensity_profiles: np.ndarray | None = None
    radial_profiles: np.ndarray | None = None
    noncircular_radial_profiles: np.ndarray | None = None
    noncircular_normalized_along_radius: np.ndarray | None = None
    angular_profiles: np.ndarray | None = None
    noncircular_angular_profiles: np.ndarray | None = None

    membrane: MembraneDetectionResult = field(default_factory=MembraneDetectionResult)
    circular_memb_localization: dict | None = None
    noncircular_memb_localization: dict | None = None
    circular_inside_intensity: np.ndarray | None = None
    noncircular_inside_intensity: np.ndarray | None = None

    comments: list[str] = field(default_factory=list)
    valid: bool = True

@dataclass(kw_only=True)
class GUV:
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


    def __post_init__(self) -> None:
        self.ves_coordinates = np.array([self.id, self.xc, self.yc, self.radius,], dtype=float)
        self.num_angles = choose_num_angles(self.radius, self.settings.profiles.target_arc_spacing)

    @property
    def full_along_radius(self) -> np.ndarray:
        profile_radius_limit = self.settings.profiles.length_excess * self.radius
        return np.arange(0, profile_radius_limit, self.settings.profiles.profile_step)

    @property
    def along_radius(self) -> np.ndarray:
        return self.full_along_radius[self.settings.profiles.pixels_to_remove:]

    @property
    def theta(self):
        return np.linspace(0, 2 * np.pi, self.num_angles, endpoint=False)

    def calculate_intensity_profiles(self) -> None: 
        intensity_profiles, self.death_mark = linear_profiles(
            self.image_view.image, 
            self.ves_coordinates, 
            self.full_along_radius,
            self.theta, self.settings.profiles)
        self.analysis.intensity_profiles = trim_central_profiles(intensity_profiles, self.settings.profiles.pixels_to_remove)

    def calculate_local_background(self) -> None:
        dilated_mask = dilate_vesicle_mask(self.image_view.global_mask, iterations=3)
        return local_background(self.image_view.image, self.ves_coordinates, dilated_mask, 
            self.settings.background.inner_margin, self.settings.background.outer_margin)

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
                self.analysis.comments.append(str(error))
                self.death_mark = True

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

    def correct_intensity_profiles(self) -> None:
        self.analysis.intensity_profiles = background_correction(self.analysis.intensity_profiles, self.analysis.background)
    def calculate_circular_radial_profile(self) -> None:
        self.analysis.radial_profiles = radial_profile(self.analysis.intensity_profiles)
    #TODO: normalize circular profiles
    def detect_circular_membrane(self) -> None:
        if self.analysis.radial_profiles is None:
            raise RuntimeError("Circular radial profile has not been calculated.")

        detection= self.analysis.membrane
        detection.peak_radius, detection.peak_index, detection.inner_border_index, detection.outer_border_index, comments, detection_failed = detect_circular_GUV(
            self.analysis.radial_profiles[:,self.settings.guv_ch], self.along_radius, self.radius, self.settings.circular_membrane)
        self.death_mark = self.death_mark or detection_failed
        self.analysis.comments.extend(comments)
        self.analysis.membrane = detection

    def detect_noncircular_membrane(self) -> None:
        if self.analysis.intensity_profiles is None:
                    raise RuntimeError("Intensity profiles have not been calculated.")
        detection= self.analysis.membrane
        detection.shape_x, detection.shape_y, detection.peak_radius_by_angle, detection.mean_radius, comments, detection_failed = detect_noncircular_GUV(
            self.analysis.intensity_profiles, self.along_radius, self.theta, self.ves_coordinates, self.settings.noncircular_membrane, self.settings.guv_ch)
        self.death_mark = self.death_mark or detection_failed
        self.analysis.comments.extend(comments)
        self.analysis.membrane = detection

    def calculate_normalized_noncircular_radial_profile(self, number_radial_points = 100) -> None:
        if self.analysis.intensity_profiles is None:
            raise RuntimeError("Intensity profiles have not been calculated.")
        if self.analysis.membrane.peak_radius_by_angle is None: 
            raise RuntimeError("Non-circular membrane not calculated.")        
        
        self.analysis.noncircular_radial_profiles, self.analysis.noncircular_normalized_along_radius, _ = normalized_radial_profile_from_detected_shape(
            self.analysis.intensity_profiles, self.along_radius, self.analysis.membrane.peak_radius_by_angle, self.settings.profiles.length_excess, number_radial_points)

    def calculate_circular_angular_profile(self) -> None: 
        if self.analysis.intensity_profiles is None:
            raise RuntimeError("Intensity profiles have not been calculated.")
        if self.analysis.membrane.inner_border_index is None: 
            raise RuntimeError("Circular membrane not calculated.")

        self.analysis.angular_profiles = angular_profile(
            self.analysis.intensity_profiles, self.analysis.membrane.inner_border_index, self.analysis.membrane.outer_border_index)

    def calculate_noncircular_angular_profile(self) -> None: 
        if self.analysis.intensity_profiles is None:
            raise RuntimeError("Intensity profiles have not been calculated.")
        if self.analysis.membrane.peak_radius_by_angle is None: 
            raise RuntimeError("Non-circular membrane not calculated.")

        self.analysis.noncircular_angular_profiles = angular_profile_from_detected_shape(
            self.analysis.intensity_profiles, self.along_radius, self.analysis.membrane.peak_radius_by_angle, self.settings.noncircular_membrane_width_pixels)

    def calculate_circular_membrane_localization(self) -> None:
        if self.analysis.intensity_profiles is None:
            raise RuntimeError("Intensity profiles have not been calculated.")
        if self.analysis.membrane.peak_index is None: 
            raise RuntimeError("Circular membrane not calculated.")
        
        self.analysis.circular_memb_localization, comment = circular_membrane_localization(
            self.analysis.intensity_profiles, self.analysis.membrane.peak_index, self.analysis.membrane.index_border_in, 
            self.analysis.membrane.index_border_out, self.settings.localization.centre_size, self.settings.guv_ch)
        self.analysis.comments.extend(comment)

    def calculate_noncircular_membrane_localization(self) -> None:
        if self.analysis.intensity_profiles is None:
            raise RuntimeError("Intensity profiles have not been calculated.")
        if self.analysis.membrane.peak_radius_by_angle is None: 
            raise RuntimeError("Noncircular membrane not calculated.")
        
        self.analysis.noncircular_memb_localization, comment = noncircular_membrane_localization(
            self.analysis.intensity_profiles, self.along_radius, self.analysis.membrane.peak_radius_by_angle, 
            self.settings.noncircular_membrane_width_pixels, self.settings.localization.centre_size, self.settings.guv_ch)
        self.analysis.comments.extend(comment)

    def calculate_circular_intensity(self) -> None:
        detection = self.analysis.membrane
        if detection.peak_radius is None:
            raise RuntimeError("Circular membrane has not been detected.")

        mask = circular_GUV_mask(self.image_view.image.shape[1:], self.xc, self.yc, detection.peak_radius)
        _, self.analysis.circular_inside_intensity = calculate_inside_intensity(self.image_view.image, mask, self.analysis.background)

    def calculate_noncircular_intensity(self) -> None:
        detection = self.analysis.membrane
        if detection.shape_x is None:
            raise RuntimeError("Noncircular membrane has not been detected.")

        mask = noncircular_GUV_mask(self.image_view.image.shape[1:], detection.shape_x, detection.shape_y)
        _, self.analysis.noncircular_inside_intensity = calculate_inside_intensity(self.image_view.image, mask, self.analysis.background)