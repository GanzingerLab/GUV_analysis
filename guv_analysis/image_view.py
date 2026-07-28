import numpy as np

class GUVImageView:
    def __init__(self, path: str, image: np.ndarray, pixel_size: float, 
                global_mask: np.ndarray | None = None, global_background: float | None = None):
        self._path = path

        self._image = image.view()
        self._image.flags.writeable = False

        self._pixel_size = pixel_size

        if global_mask is None:
            self._global_mask = None
        else:
            self._global_mask = global_mask.view()
            self._global_mask.flags.writeable = False

        self._global_background = global_background

    @property
    def path(self) -> str:
        return self._path

    @property
    def image(self) -> np.ndarray:
        return self._image

    @property
    def pixel_size(self) -> float:
        return self._pixel_size

    @property
    def global_mask(self) -> np.ndarray | None:
        return self._global_mask

    @property
    def global_background(self) -> float | None:
        return self._global_background
