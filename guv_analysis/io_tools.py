import os
from pathlib import Path
import numpy as np
import pandas as pd
from bioio import BioImage
import tifffile as tif
import json

#TODO: consider time usage (more dimensions in the array)
def open_czi(path):
    img_bio = BioImage(path)
    img = img_bio.get_image_data()
    img, dims = squeeze_singleton_dims(img, img_bio.dims.order)
    return img

def open_tif(path):
    return tif.imread(path)

def open_image(path):
    if path[-4:] == '.tif':
        return open_tif(path)
    if path[-4:] == '.czi':
        return open_czi(path)

def open_data(path, detection_suffix):
    img = open_image(path)
    img = ensure_channel_axis(img)
    detection = open_detection(path, detection_suffix)
    return img, detection

def ensure_channel_axis(img):
    if img.ndim == 2:
        img = img[np.newaxis, :, :]

    return img

def squeeze_singleton_dims(img, dims):
    keep_dims = []

    for size, dim in zip(img.shape, dims):
        if size != 1:
            keep_dims.append(dim)

    img_squeezed = np.squeeze(
        img,
        axis=tuple(axis for axis, size in enumerate(img.shape) if size == 1)
    )

    dims_squeezed = "".join(keep_dims)

    return img_squeezed, dims_squeezed

def get_output_folder(image_path, base_path):
    """
    Create and return the output folder corresponding to one image.

    Example
    -------
    base_path:
        D:\\Data\\EVOLF

    image_path:
        D:\\Data\\EVOLF\\a\\b\\file.czi

    returns:
        D:\\Data\\EVOLF\\output\\a\\b
    """

    relative_path = os.path.relpath(image_path, base_path)
    relative_folder = os.path.dirname(relative_path)

    output_folder = os.path.join(base_path, "output", relative_folder)
    os.makedirs(output_folder, exist_ok=True)

    return output_folder

def open_detection(image_path, detection_suffix):
    image_path = Path(image_path)
    csv_path = image_path.with_name(f"{image_path.stem}{detection_suffix}")
    
    if not os.path.exists(csv_path):
        print(f"Detection CSV not found: {csv_path}")
        return None
        

    locs = pd.read_csv(csv_path, index_col=False).to_numpy()

    return locs

def save_shape_normalized_profiles(
    output_folder,
    image_path,
    profiles_data
):
    """
    Save shape-normalized radial and angular profiles for all GUVs in one source image.

    The NPZ file contains:
        radial profile per vesicle
        angular profile per vesicle
        normalized distance axis per vesicle
    """

    image_name = os.path.basename(image_path)
    image_stem = os.path.splitext(image_name)[0]

    profile_path = os.path.join(
        output_folder,
        f"{image_stem}_shape_normalized_profiles.npz"
    )

    np.savez_compressed(profile_path, **profiles_data)

    return profile_path

def save_membrane_positions(
    output_folder,
    image_path,
    membrane_positions
):
    """
    Save detected membrane positions for all GUVs in one source image.

    The JSON contains one entry per vesicle id.
    Each vesicle contains a list of [x, y] membrane coordinates.
    """

    image_name = os.path.basename(image_path)
    image_stem = os.path.splitext(image_name)[0]

    membrane_data = {
        "source_file": image_name,
        "vesicles": membrane_positions
    }

    membrane_path = os.path.join(
        output_folder,
        f"{image_stem}_membrane_positions.json"
    )

    with open(membrane_path, "w") as f:
        json.dump(membrane_data, f, indent=4)

    return membrane_path

def reorder_summary_columns(results_df):
    """
    Reorder summary table columns so paths and main identifiers appear first.
    """

    first_columns = [
        "image_path",
        "membrane_position_file",
        "profile_data_file",
        "vesicle_id",
        "xc",
        "yc",
        "radius",
    ]

    first_columns = [
        col for col in first_columns
        if col in results_df.columns
    ]

    other_columns = [
        col for col in results_df.columns
        if col not in first_columns
    ]

    results_df = results_df[
        first_columns + other_columns
    ]

    return results_df

def load_membrane_positions_json(membrane_path):
    """
    Load detected membrane positions from one JSON file.

    Returns
    -------
    source_file : str
        Name of the original image file.

    vesicles : dict
        Dictionary where each key is a vesicle id and each value is a
        NumPy array with shape:
            num_angles x 2

        Column 0 is x.
        Column 1 is y.
    """

    with open(membrane_path, "r") as f:
        membrane_data = json.load(f)

    vesicles = {
        vesicle_id: np.array(membrane_xy)
        for vesicle_id, membrane_xy in membrane_data["vesicles"].items()
    }

    return membrane_data["source_file"], vesicles