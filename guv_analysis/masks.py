import numpy as np
from skimage.draw import polygon2mask

def circular_GUV_mask(image_shape, xc, yc, peak_radius):
    """Create a circular interior mask using the detected membrane radius."""
    height, width = image_shape
    yy, xx = np.ogrid[:height, :width]
    return (xx - xc)**2 + (yy - yc)**2 <= peak_radius**2

def noncircular_GUV_mask(image_shape, shape_x, shape_y):
    """Create an interior mask from the detected noncircular membrane contour."""
    polygon_coordinates = np.column_stack((shape_y, shape_x))
    return polygon2mask(image_shape, polygon_coordinates)