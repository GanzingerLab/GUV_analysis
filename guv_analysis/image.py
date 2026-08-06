import numpy as np 
from .io_tools import open_data
from .GUV import GUV
from .background import mask_all_vesicles
from .image_view import GUVImageView
from .background import dilate_vesicle_mask, global_background
from .filter import filter_clumped_vesicles

class GUVImage:
    def __init__(self, path, settings):
        self.path = path
        self.settings = settings
        self.image, self.pixel_size , self.detections= open_data(path, settings.detection_suffix) 
        self.image_view = GUVImageView(path = self.path, image = self.image, pixel_size = self.pixel_size, global_mask = None)

        self.guvs = {int(guv_id): GUV(id=int(guv_id), xc=float(xc), yc=float(yc), radius=float(radius), image_view = self.image_view, settings = settings, on_death_marked=self._on_guv_death_marked)
            for guv_id, xc, yc, radius in self.detections}
        self.bad_GUVs: set[int] = set()
        self.good_GUVs: set[int] = set(self.guvs.keys())
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
    def filter_GUVs_in_clusters(self):
        if self.global_mask is None:
            self.make_global_mask()
        dilated_mask = dilate_vesicle_mask(self.global_mask, iterations=5)
        clumped_vesicles = filter_clumped_vesicles(self.detections, dilated_mask)
        for i in clumped_vesicles:
            self.guvs[i].death_mark = True
            self.guvs[i].analysis.comments.append("GUV in cluster")
        self.bad_GUVs.update(clumped_vesicles)
        self.good_GUVs = self.good_GUVs - self.bad_GUVs 
    def kill_guvs_on_comment(self):
        for guv_id in list(self.good_GUVs): 
            guv = self.guvs[guv_id]
            if any(comment in guv.analysis.comments for comment in self.settings.filter.killing_comments):
                guv.mark_dead()

    def _on_guv_death_marked(self, guv: GUV) -> None:
        self.good_GUVs.discard(guv.id)
        self.bad_GUVs.add(guv.id)
        






