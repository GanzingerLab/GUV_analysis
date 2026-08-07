import numpy as np 
import pandas as pd

from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any
import h5py

from .io_tools import open_data
from .GUV import GUV
from .background import mask_all_vesicles
from .image_view import GUVImageView
from .background import dilate_vesicle_mask, global_background
from .filter import filter_clumped_vesicles
from .hdf_manager import write_hdf5_value

class GUVImage:
    """
    Container for one microscopy image and all detected GUVs in that image.

    This class loads the image, loads the detection CSV, creates one GUV object
    per detected vesicle, and keeps track of good and bad GUVs during analysis.

    Main attributes
    ---------------
    image : np.ndarray
        Image array with shape channels x y x.

    detections : np.ndarray
        Detection table. Each row contains:
            vesicle_id, xc, yc, radius

    guvs : dict[int, GUV]
        Dictionary containing one GUV object per detected vesicle.

    good_GUVs : set[int]
        IDs of GUVs that have not been death-marked.

    bad_GUVs : set[int]
        IDs of GUVs that have been death-marked.
    """
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

    def extract_parameter(self, parameter: str, GUVs="all") -> dict[int, Any]:
        """Extract one parameter from selected GUVs."""
        if GUVs == "all":
            guv_ids = self.guvs.keys()
        elif GUVs == "good":
            guv_ids = self.good_GUVs
        elif GUVs == "bad":
            guv_ids = self.bad_GUVs
        else:
            guv_ids = GUVs

        results = {}

        for guv_id in guv_ids:
            value = self.guvs[guv_id]
            try:
                for attribute in parameter.split("."):
                    value = getattr(value, attribute)
            except AttributeError as error:
                raise AttributeError(f"GUV {guv_id} has no parameter {parameter!r}.") from error

            if value is None:
                continue

            results[guv_id] = value

        return results
    def save_parameter_hdf5(self, path: str | Path, parameter: str, GUVs="all") -> None:
        """Save one parameter from selected GUVs into an HDF5 file."""
        results = self.extract_parameter(parameter, GUVs)
        path = Path(path)

        with h5py.File(path, "w") as file:
            file.attrs["parameter"] = parameter
            file.attrs["format_version"] = 1

            results_group = file.create_group("results")

            for guv_id, value in results.items():
                guv_group = results_group.create_group(str(guv_id))
                write_hdf5_value(guv_group, "value", value)

    def save_hdf5(self, path: str | Path, guvs="all") -> None:
        """Save selected GUV analysis results to an HDF5 file."""
        from .hdf_manager import save_full_hdf5

        save_full_hdf5(self, path, guvs)

    def _on_guv_death_marked(self, guv: GUV) -> None:
        self.good_GUVs.discard(guv.id)
        self.bad_GUVs.add(guv.id)
        






