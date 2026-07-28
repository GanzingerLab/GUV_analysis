import numpy as np 
from .io_tools import open_data
from .GUV import GUV
from .background import mask_all_vesicles
from .image_view import GUVImageView
from .background import dilate_vesicle_mask, global_background

class GUVImage:
    def __init__(self, path, settings):
        self.path = path
        self.settings = settings
        self.image, self.pixel_size , self.detections= open_data(path, settings.detection_suffix) 
        self.image_view = GUVImageView(path = self.path, image = self.image, pixel_size = self.pixel_size, global_mask = None)

        self.guvs = {int(guv_id): GUV(id=int(guv_id), xc=float(xc), yc=float(yc), radius=float(radius), image_view = self.image_view, settings = settings)
            for guv_id, xc, yc, radius in self.detections}

        self.global_mask = None
        self.global_background = None
    def make_global_mask(self):
        self.global_mask = mask_all_vesicles(self.image, self.detections)
        self.image_view._global_mask = self.global_mask
    def calculate_global_background(self):
        if self.global_mask is None:
            self.make_global_mask()
        dilated_mask = dilate_vesicle_mask(self.global_mask)
        self.global_background = global_background(self.image, dilated_mask)
        self.image_view._global_background = self.global_background





