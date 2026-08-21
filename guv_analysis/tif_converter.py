from guv_analysis.io_tools import squeeze_singleton_dims
from pathlib import Path
from bioio import BioImage
from pathlib import Path
import tifffile
import numpy as np


def clean_tag(tag):
    """Remove XML namespace."""
    return tag.split("}")[-1]


def extract_timestamps(root):
    """
    Best-effort extraction of actual time points from CZI XML.

    Returns timestamps in acquisition order when available.
    """
    timestamps = []

    for elem in root.iter():
        tag = clean_tag(elem.tag)

        if tag in {"TimeStamp", "Timestamp", "DeltaT"}:
            if elem.text and elem.text.strip():
                try:
                    timestamps.append(float(elem.text.strip()))
                except ValueError:
                    timestamps.append(elem.text.strip())

    return timestamps

def find_elements(root, tag):
    """Find every XML element with a given local tag name."""
    return [
        elem
        for elem in root.iter()
        if clean_tag(elem.tag) == tag
    ]


def element_text(elem, child_name):
    """Get text of a direct child, ignoring namespaces."""
    for child in elem:
        if clean_tag(child.tag) == child_name:
            return child.text
    return None


def build_imagej_compatible_metadata(img, data, axes):
    root = img.metadata
    lines = []

    # -----------------------------
    # Basic image info
    # -----------------------------
    lines.append(f"BitsPerPixel = {data.dtype.itemsize * 8}")
    lines.append("DimensionOrder = XYCZT")
    lines.append(f"PixelType = {data.dtype.name}")

    for dim in "CTXYZ":
        if dim in img.dims.order:
            lines.append(f"Size{dim} = {getattr(img.dims, dim)}")

    # -----------------------------
    # Channels
    # -----------------------------
    display_channels = []

    for elem in find_elements(root, "Channels"):
        channels = [
            child for child in elem
            if clean_tag(child.tag) == "Channel"
        ]

        if len(channels) == img.dims.C:
            display_channels = channels
            break

    if display_channels:
        properties = {
            "Name": "Information|Image|Channel|Name",
            "Color": "Information|Image|Channel|Color",
            "DyeName": "Information|Image|Channel|Fluor",
            "DyeMaxExcitation":
                "Information|Image|Channel|ExcitationWavelength",
            "DyeMaxEmission":
                "Information|Image|Channel|EmissionWavelength",
        }

        for xml_name, output_name in properties.items():
            for i, channel in enumerate(display_channels, start=1):

                if xml_name == "Name":
                    value = channel.attrib.get("Name")
                else:
                    value = element_text(channel, xml_name)

                if value is not None:
                    lines.append(
                        f"{output_name} #{i} = {value}"
                    )

    # -----------------------------
    # Pixel calibration
    # -----------------------------
    px = img.physical_pixel_sizes.X
    py = img.physical_pixel_sizes.Y
    pz = img.physical_pixel_sizes.Z

    scaling_index = 1

    if px is not None:
        lines.append(
            f"Scaling|Distance|DefaultUnitFormat #{scaling_index} = µm"
        )
        lines.append(
            f"Scaling|Distance|Id #{scaling_index} = X"
        )
        lines.append(
            f"Scaling|Distance|Value #{scaling_index} = {px * 1e-6:.15E}"
        )
        scaling_index += 1

    if py is not None:
        lines.append(
            f"Scaling|Distance|DefaultUnitFormat #{scaling_index} = µm"
        )
        lines.append(
            f"Scaling|Distance|Id #{scaling_index} = Y"
        )
        lines.append(
            f"Scaling|Distance|Value #{scaling_index} = {py * 1e-6:.15E}"
        )
        scaling_index += 1

    if pz is not None and img.dims.Z > 1:
        lines.append(
            f"Scaling|Distance|DefaultUnitFormat #{scaling_index} = µm"
        )
        lines.append(
            f"Scaling|Distance|Id #{scaling_index} = Z"
        )
        lines.append(
            f"Scaling|Distance|Value #{scaling_index} = {pz * 1e-6:.15E}"
        )

    # -----------------------------
    # Useful single-value metadata
    # -----------------------------
    wanted_single = {
        "FrameRateTarget":
            "Experiment|AcquisitionBlock|AcquisitionModeSetup|Detector|FrameRateTarget",

        "ObjectiveName":
            "Scaling|AutoScaling|ObjectiveName",

        "ExposureTime":
            "Experiment|AcquisitionBlock|AcquisitionModeSetup|Detector|ExposureTime",
    }

    for xml_tag, output_key in wanted_single.items():
        elems = find_elements(root, xml_tag)

        for elem in elems:
            if elem.text and elem.text.strip():
                lines.append(
                    f"{output_key} = {elem.text.strip()}"
                )
                break

    # -----------------------------
    # Channel-specific quantitative metadata
    # -----------------------------
    channel_keys = {
        "PinholeSize":
            "Information|Image|Channel|PinholeSize",

        "Voltage":
            "Information|Image|Channel|Voltage",

        "Wavelength":
            "Information|Image|Channel|Wavelength",

        "Transmission":
            "Information|Image|Channel|Transmission",

        "DigitalGain":
            "Information|Image|Channel|DigitalGain",

        "DetectorMode":
            "Information|Image|Channel|DetectorMode",
    }

    # Search all elements and collect values in XML order.
    for xml_tag, output_key in channel_keys.items():
        values = []

        for elem in find_elements(root, xml_tag):
            if elem.text and elem.text.strip():
                values.append(elem.text.strip())

        # Keep only one value per actual image channel.
        # This avoids dumping unused detector/channel definitions.
        if len(values) >= img.dims.C:
            values = values[:img.dims.C]

        for i, value in enumerate(values, start=1):
            lines.append(
                f"{output_key} #{i} = {value}"
            )

    # -----------------------------
    # Actual time-series timing
    # -----------------------------
    if img.dims.T > 1:
        timestamps = extract_timestamps(root)

        if timestamps:
            for i, t in enumerate(timestamps, start=1):
                lines.append(
                    f"Information|Image|T|TimeStamp #{i} = {t}"
                )

            # If timestamps are numeric seconds, we can also calculate
            # the average actual frame interval.
            try:
                ts = np.asarray(timestamps, dtype=float)

                if len(ts) > 1:
                    intervals = np.diff(ts)

                    lines.append(
                        f"Information|Image|T|MeanFrameInterval = "
                        f"{intervals.mean()}"
                    )

                    lines.append(
                        f"Information|Image|T|MeanFrameRate = "
                        f"{1 / intervals.mean()}"
                    )

            except (TypeError, ValueError):
                pass

    return "\n".join(lines)



def czi_to_tif(czi_path, save_path = None):
    """
    Convert a Zeiss CZI image to an ImageJ-compatible TIFF.

    A curated subset of the original CZI metadata is converted to the
    ImageJ/Bio-Formats-style ``key = value`` format so that the resulting
    TIFF remains compatible with metadata in TIFF files saved through Fiji/ImageJ.

    Preserved metadata includes, when available:
        - image dimensions and dimension order
        - pixel type and bit depth
        - physical pixel size in X and Y
        - Z spacing for Z-stacks
        - channel names
        - channel colors
        - fluorophore names
        - excitation and emission wavelengths
        - objective information
        - frame-rate target
        - exposure time
        - pinhole size
        - detector voltage
        - laser wavelength
        - laser transmission
        - digital gain
        - detector mode

    Parameters
    ----------
    czi_path : str or pathlib.Path
        Path to the input CZI file.

    save_path : str or pathlib.Path, optional
        Folder in which to save the converted TIFF. If None, the TIFF is
        saved next to the original CZI file. The output filename is the same
        as the CZI filename, with the extension changed to ``.tif``.

    Returns
    -------
    pathlib.Path
        Path to the saved TIFF file.
    """
    czi_path = Path(czi_path)

    img = BioImage(czi_path)

    # Full BioIO array
    data = img.data
    dims = img.dims.order

    # Remove singleton T/C/Z/etc. dimensions
    data, axes = squeeze_singleton_dims(data, dims)

    px = img.physical_pixel_sizes.X
    py = img.physical_pixel_sizes.Y

    info = build_imagej_compatible_metadata(
        img,
        data,
        axes,
    )

    if save_path is None:
        output = czi_path.with_suffix(".tif")

    else:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        name = czi_path.with_suffix(".tif").name
        output = Path(save_path) / name

    tifffile.imwrite(
        output,
        data,
        imagej=True,
        resolution=(1 / px, 1 / py),
        metadata={"axes": axes, "unit": "um", "Info": info})
    # print("Original:", img.dims.order, img.shape)
    # print("Writing:", axes, data.shape)
    return output