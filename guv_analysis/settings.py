from dataclasses import dataclass, field, asdict
from pprint import pformat

@dataclass(frozen=True)
class BackgroundSettings:
    method: str = "local_then_global"
    inner_margin: float = 1.2
    outer_margin: float = 2.0
    mask_dilation_iterations: int = 3
    minimum_pixels: int = 6


@dataclass(frozen=True)
class ProfileSettings:
    target_arc_spacing: int = 1.1
    profile_step: float = 1.0
    length_excess: float = 1.5
    smoothing_window: int = 5
    pixels_to_remove: int = 2
    noncircular_membrane_width_pixels: int = 3


@dataclass(frozen=True)
class CircularMembraneSettings:
    peak_height: float = 0.1
    peak_distance: int = 5
    peak_prominence: float = 0.15
    width_relative_height: float = 0.5
    wide_peak_fraction: float = 0.25
    inside_signal_fraction: float = 0.30
    outside_signal_fraction: float = 0.40


@dataclass(frozen=True)
class NonCircularMembraneSettings:
    profile_smoothing_window: int = 7
    contour_smoothing_window: int = 5
    smoothness_weight: float = 0.4
    radius_prior_weight: float = 0.1
    min_radius_fraction: float = 0.35
    max_radius_fraction: float = 1.5
    max_single_jump_fraction: float = 0.10
    max_contour_variation_fraction: float = 0.03


@dataclass(frozen=True)
class LocalizationSettings:
    centre_size: int = 0.25


@dataclass(frozen=True)
class PlotSettings:
    crop_size_factor: float = 1.5
    dpi: int = 300
    show: bool = True


@dataclass(frozen=True)
class AnalysisSettings:
    guv_ch:int  = 0
    detection_suffix: str = "_detected_vesicles.csv"
    background: BackgroundSettings = field(
        default_factory=BackgroundSettings
    )
    profiles: ProfileSettings = field(
        default_factory=ProfileSettings
    )
    circular_membrane: CircularMembraneSettings = field(
        default_factory=CircularMembraneSettings
    )
    noncircular_membrane: NonCircularMembraneSettings = field(
        default_factory=NonCircularMembraneSettings
    )
    localization: LocalizationSettings = field(
        default_factory=LocalizationSettings
    )
    plotting: PlotSettings = field(
        default_factory=PlotSettings
    )

    def show(self) -> None:
        print(pformat(asdict(self), sort_dicts=False))