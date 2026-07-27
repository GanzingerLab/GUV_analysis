import numpy as np 

def crop_around_guv(img, size_factor, ves_coordinates, return_origin=False):
    """
    Crop all image channels around one GUV, shifting the crop when necessary
    so it remains inside the image boundaries.

    Parameters
    ----------
    img : np.ndarray
        Channel-first image with shape (n_channels, height, width).

    size_factor : float
        Crop half-width expressed as a multiple of the GUV radius.

    ves_coordinates : array-like
        [vesicle_id, xc, yc, radius].

    Returns
    -------
    cropped_img : np.ndarray
        Cropped channel-first image with shape
        (n_channels, crop_height, crop_width).
    """
    xc = int(round(ves_coordinates[1]))
    yc = int(round(ves_coordinates[2]))
    radius = ves_coordinates[3]

    _, height, width = img.shape

    half_size = int(np.ceil(size_factor * radius))
    crop_size = 2 * half_size

    crop_width = min(crop_size, width)
    crop_height = min(crop_size, height)

    x_start = xc - half_size
    y_start = yc - half_size

    x_start = min(max(x_start, 0), width - crop_width)
    y_start = min(max(y_start, 0), height - crop_height)

    x_end = x_start + crop_width
    y_end = y_start + crop_height

    cropped_img = img[:, y_start:y_end, x_start:x_end]

    if return_origin:
        return cropped_img, (x_start, y_start)

    return cropped_img