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
from guv_analysis.profiles import angular_profile, linear_profiles, radial_profile, trim_central_profiles, choose_num_angles
from guv_analysis.signal_quantification import localization
from guv_analysis.settings import AnalysisSettings
from guv_analysis.image_view import GUVImageView




@dataclass
class MembraneDetectionResult:
    comments: list[str] = field(default_factory=list)

    peak_radius: float | None = None
    inner_border_index: int | None = None
    outer_border_index: int | None = None

    peak_positions: np.ndarray | None = None
    shape_x: np.ndarray | None = None
    shape_y: np.ndarray | None = None
    

@dataclass
class GUVAnalysis:
    background: np.ndarray | None = None
    background_method: str | None = None

    intensity_profiles: np.ndarray | None = None
    radial_profiles: np.ndarray | None = None
    angular_profiles: np.ndarray | None = None

    membrane: MembraneDetectionResult = field(default_factory=MembraneDetectionResult)
    localization: np.ndarray | None = None

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
        self.analysis.intensity_profiles = background_correction(
                    self.analysis.intensity_profiles,
                    self.analysis.background
                )
    def calculate_circular_radial_profile(self) -> None:
        self.analysis.radial_profiles = radial_profile(self.analysis.intensity_profiles)
    #TODO: normalize circular profiles
    def detect_circular_membrane(self) -> None:
        if self.analysis.radial_profiles is None:
            raise RuntimeError("Circular radial profile has not been calculated.")

        detection= self.analysis.membrane
        detection.peak_radius, detection.inner_border_index, detection.outer_border_index, comments, self.death_mark = detect_circular_GUV(self.analysis.radial_profiles, self.radius, self.settings.circular_membrane)
        self.analysis.comments.extend(comments)
        self.analysis.membrane = detection

    # def detect_noncircular_membrane(self) -> None:



        
