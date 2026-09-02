from dataclasses import dataclass, field, asdict
from pprint import pformat


@dataclass()
class BackgroundSettings:
    """Settings for background estimation."""

    method: str = "local_then_global"  # Options: "local", "global", "local_then_global"
    inner_margin: float = 1.2  # Inner radius of local background annulus, relative to GUV radius
    outer_margin: float = 2.0  # Outer radius of local background annulus, relative to GUV radius
    mask_dilation_iterations: int = 3  # Mask expansion around GUVs for global background estimation
    minimum_pixels: int = 6  # Minimum pixels needed to accept a background estimate


@dataclass()
class ProfileSettings:
    """Settings for radial and angular intensity profiles."""

    flatten_angular_profiles: bool = True  # If True, correct angular profiles for broad intensity trends
    target_arc_spacing: float = 1.1  # Approximate spacing between angular profiles at the membrane. Later, this is used to claculate the number of angles needed to reach a separation of the number set here of pixels between angles. 
    profile_step: float = 1.0  # Radial sampling step in pixels -> Step size for intensity profiles. 
    length_excess: float = 1.5 # Radial profile length relative to GUV radius from disGUVery. 
    minimum_outer_padding_pixels: float =7.0 # Minimum radial distance sampled beyond the approximate DisGUVery radius.
    edge_proximity: float = 1.05  # Only used to check whether the GUV membrane itself is too close to the image edge
    smoothing_window: int = 5  # Angular smoothing window for intensity profiles. odd number. 
    pixels_to_remove: int = 2  # Central radial samples removed; equals pixels even if profile_step != 1.
    noncircular_membrane_width_pixels: int = 3  # Expected membrane width in pixels for non-circular membrane detection. Must be a positive odd number.

    min_radial_membrane_support: float = 3.0  # Minimum membrane_prominence / local_noise required for one angle to be considered membrane. The proportion of angles considered membranes is the fraction_membrane
    min_fraction_membrane: float = 0.6  # Minimum membrane fraction required to keep the GUV
    min_angular_variation: float = 0.2  # Minimum angular variation required after flattening
    max_angular_fit_error: float = 0.4  # Maximum accepted angular flattening fit error
    min_angular_fit_correlation: float = 0.6  # Minimum accepted angular flattening fit correlation


@dataclass()
class CircularMembraneSettings:
    """Settings for circular membrane detection."""
    peak_height: float = 0.1  # Minimum relative peak height. Default is 0.1, meaning the peak must be at least 10% higher than the minimum value.
    peak_distance: int = 5  # Minimum distance between candidate peaks, in radial samples
    peak_prominence: float = 0.15  # Minimum peak prominence. Default is 0.15, meaning the peak must be at least 15% higher than the highest surrounding baseline.
    width_relative_height: float = 0.5  # Relative height used to measure peak width
    wide_peak_fraction: float = 1/5  # Maximum peak width relative to radius before flagging as wide
    inside_signal_fraction: float = 0.30  # Threshold for high signal inside the GUV
    outside_signal_fraction: float = 0.40  # Threshold for high signal outside the GUV


@dataclass()
class NonCircularMembraneSettings:
    """Settings for non-circular membrane detection."""

    profile_smoothing_window: int = 7  # Smoothing window before contour detection
    contour_smoothing_window: int = 5  # Smoothing window for final contour (localizations are smoothed after detection)
    smoothness_weight: float = 0.4  # Higher values force smoother contours
    radius_prior_weight: float = 0.1  # Higher values keep contour closer to approximate radius
    min_radius_fraction: float = 0.35  # Minimum allowed radius as fraction of approximate radius (disGUVEry radius)
    max_radius_fraction: float = 1.5  # Maximum allowed radius as fraction of approximate radius (disGUVEry radius)
    max_single_jump_fraction: float = 0.20  # Maximum allowed local radius jump before flagging contour_jump
    max_contour_variation_fraction: float = 0.05  # Maximum allowed contour irregularity before flagging irregular_contour


@dataclass()
class LocalizationSettings:
    """Settings for localization measurements."""
    centre_size: float = 0.5  # proportion of pixels from the center of the profile considered to be center to calculate localization. 

@dataclass()
class PlotSettings:
    """Settings for plotting."""
    #TODO: implememnt these settings in the plotting functions... 
    crop_size_factor: float = 1.5  # Crop size around GUV, relative to radius
    dpi: int = 300  # Resolution for saved plots 
    show: bool = True  # If True, show plots interactively; if False, only save/close them


@dataclass
class FilterSettings:
    """Settings defining which comments kill a GUV."""

    killing_comments: list[str] = field(default_factory=lambda: [ # Comments that mark a GUV as bad (dead)
        "contour_jump",
        "irregular_contour"
    ]) 

    surviving_comments: list[str] = field(default_factory=lambda: [# Comments that are kept as warnings but do not kill the GUV
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
        """Check that comments are not both killing and surviving."""

        overlap = set(self.killing_comments) & set(self.surviving_comments)

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
    """Main settings object for the full GUV analysis."""

    guv_ch: int = 0  # Channel containing the GUV membrane marker; channel numbering starts at 0
    skip_death_marked: bool = True  # If True, skip later analysis steps after a GUV is marked dead
    detection_suffix: str = "_detected_vesicles.csv"  # Suffix of detection CSV linked to each image

    background: BackgroundSettings = field(default_factory=BackgroundSettings)  # Background estimation settings
    profiles: ProfileSettings = field(default_factory=ProfileSettings)  # Profile extraction settings
    circular_membrane: CircularMembraneSettings = field(default_factory=CircularMembraneSettings)  # Circular detection settings
    noncircular_membrane: NonCircularMembraneSettings = field(default_factory=NonCircularMembraneSettings)  # Non-circular detection settings
    localization: LocalizationSettings = field(default_factory=LocalizationSettings)  # Localization settings
    plotting: PlotSettings = field(default_factory=PlotSettings)  # Plotting settings
    filter: FilterSettings = field(default_factory=FilterSettings)  # Comment/filter settings

    def show(self) -> None:
        """Print all current settings."""

        print(pformat(asdict(self), sort_dicts=False))