import os
from pathlib import Path
import numpy as np
import pandas as pd
from bioio import BioImage
import tifffile as tif

#TODO: consider time usage 
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
    img = tif.imread(path)
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