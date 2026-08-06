from dataclasses import dataclass, field, asdict
from pprint import pformat

@dataclass()
class BackgroundSettings:
    method: str = "local_then_global"
    inner_margin: float = 1.2
    outer_margin: float = 2.0
    mask_dilation_iterations: int = 3
    minimum_pixels: int = 6


@dataclass()
class ProfileSettings:
    flatten_angular_profiles: bool = True
    target_arc_spacing: int = 1.1
    profile_step: float = 1.0
    length_excess: float = 2.0
    smoothing_window: int = 5
    pixels_to_remove: int = 2
    noncircular_membrane_width_pixels: int = 5
    min_radial_membrane_support: float = 3.0
    min_fraction_membrane: float = 0.6
    min_angular_variation: float = 0.2
    max_angular_fit_error: float = 0.4
    min_angular_fit_correlation: float = 0.6



@dataclass()
class CircularMembraneSettings:
    peak_height: float = 0.1
    peak_distance: int = 5
    peak_prominence: float = 0.15
    width_relative_height: float = 0.5
    wide_peak_fraction: float = 1/3
    inside_signal_fraction: float = 0.30
    outside_signal_fraction: float = 0.40


@dataclass()
class NonCircularMembraneSettings:
    profile_smoothing_window: int = 7
    contour_smoothing_window: int = 5
    smoothness_weight: float = 0.4
    radius_prior_weight: float = 0.1
    min_radius_fraction: float = 0.35
    max_radius_fraction: float = 1.5
    max_single_jump_fraction: float = 0.10
    max_contour_variation_fraction: float = 0.03


@dataclass()
class LocalizationSettings:
    centre_size: int = 0.25


@dataclass()
class PlotSettings:
    crop_size_factor: float = 1.5
    dpi: int = 300
    show: bool = True

@dataclass
class FilterSettings:
    killing_comments: list[str] = field(default_factory=lambda: [
        "contour_jump",
        "irregular_contour"
    ])

    surviving_comments: list[str] = field(default_factory=lambda: [
        "confetti",
        "invalid_polarization_fit",
        "wide_membrane",
        "high_int_inside",
        "high_int_outside",
        "angular_intensity_variation_too_low",
        "angular_intensity_fit_error_too_high",
        "angular_intensity_fit_correlation_too_low",
        "high angular error",
    ])
    def __post_init__(self) -> None:
        overlap = set(self.killing_comments) & set(
            self.surviving_comments
        )

        if overlap:
            raise ValueError(
                "Comments cannot be both killing and surviving: "
                f"{sorted(overlap)}"
            )

    def move_to_killing(self, comment: str) -> None:
        """Move a comment from surviving to killing."""
        while comment in self.surviving_comments:
            self.surviving_comments.remove(comment)

        if comment not in self.killing_comments:
            self.killing_comments.append(comment)

    def move_to_surviving(self, comment: str) -> None:
        """Move a comment from killing to surviving."""
        while comment in self.killing_comments:
            self.killing_comments.remove(comment)

        if comment not in self.surviving_comments:
            self.surviving_comments.append(comment)


@dataclass()
class AnalysisSettings:
    guv_ch:int  = 0
    skip_death_marked: bool = True
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
    filter: FilterSettings = field(
            default_factory=FilterSettings
        )

    def show(self) -> None:
        print(pformat(asdict(self), sort_dicts=False))