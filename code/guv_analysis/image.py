def crop_around_guv(img, size_factor, ves_coordinates):
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
    xc = ves_coordinates[1]
    yc = ves_coordinates[2]
    radius = ves_coordinates[3]

    _, height, width = img.shape
    half_size = int(size_factor * radius)

    x_start = int(xc - half_size)
    x_end = int(xc + half_size)
    y_start = int(yc - half_size)
    y_end = int(yc + half_size)

    # Shift the crop instead of reducing its size near a boundary.
    if x_start < 0:
        x_end -= x_start
        x_start = 0

    if x_end > width:
        x_start -= x_end - width
        x_end = width

    if y_start < 0:
        y_end -= y_start
        y_start = 0

    if y_end > height:
        y_start -= y_end - height
        y_end = height

    # Protect against a requested crop larger than the complete image.
    x_start = max(0, x_start)
    y_start = max(0, y_start)

    return img[:, y_start:y_end, x_start:x_end]