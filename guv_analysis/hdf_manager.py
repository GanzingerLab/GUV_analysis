from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .settings import AnalysisSettings


def write_hdf5_value(group, name: str, value: Any) -> None:
    """
    Recursively save a Python value inside an HDF5 group.

    Dataclasses and dictionaries are stored as HDF5 groups. NumPy arrays
    and homogeneous collections are stored as datasets. Scalars are stored
    as attributes. None values are represented by a Boolean marker.

    Parameters
    ----------
    group : h5py.Group or h5py.File
        HDF5 location in which the value will be stored.

    name : str
        Name of the created attribute, dataset, or subgroup.

    value : Any
        Python value to save.

    Raises
    ------
    TypeError
        If the value has an unsupported type.
    """
    if value is None:
        group.attrs[f"{name}__is_none"] = True
        return

    if is_dataclass(value):
        subgroup = group.create_group(name)
        for item in fields(value):
            write_hdf5_value(subgroup, item.name, getattr(value, item.name))
        return

    if isinstance(value, dict):
        subgroup = group.create_group(name)
        for key, item in value.items():
            write_hdf5_value(subgroup, str(key), item)
        return

    if isinstance(value, np.ndarray):
        compression = "gzip" if value.ndim > 0 and value.size > 0 else None
        group.create_dataset(name, data=value, compression=compression)
        return

    if isinstance(value, (list, tuple, set)):
        values = list(value)

        if not values:
            dtype = h5py.string_dtype(encoding="utf-8")
            dataset = group.create_dataset(name, data=np.asarray([], dtype=dtype))
            dataset.attrs["collection_type"] = type(value).__name__
            return

        if all(isinstance(item, str) for item in values):
            dtype = h5py.string_dtype(encoding="utf-8")
            dataset = group.create_dataset(name, data=np.asarray(values, dtype=dtype))
            dataset.attrs["collection_type"] = type(value).__name__
            return

        array = np.asarray(values)

        if array.dtype != object:
            compression = "gzip" if array.ndim > 0 and array.size > 0 else None
            dataset = group.create_dataset(name, data=array, compression=compression)
            dataset.attrs["collection_type"] = type(value).__name__
            return

        subgroup = group.create_group(name)
        subgroup.attrs["collection_type"] = type(value).__name__

        for index, item in enumerate(values):
            write_hdf5_value(subgroup, str(index), item)

        return

    if isinstance(value, Path):
        group.attrs[name] = str(value)
        return

    if isinstance(value, (str, int, float, bool, np.generic)):
        group.attrs[name] = value
        return

    raise TypeError(f"Cannot save {name!r}: unsupported type {type(value).__name__}")


def read_hdf5_value(group, name: str, current_value=None):
    """
    Restore one Python value from an HDF5 group.

    The function reverses the structure created by ``write_hdf5_value``.
    Existing dataclass instances are updated recursively, while datasets,
    dictionaries, collections, scalars, and None values are reconstructed.

    Parameters
    ----------
    group : h5py.Group or h5py.File
        HDF5 location containing the saved value.

    name : str
        Name of the attribute, dataset, or subgroup to read.

    current_value : Any, optional
        Existing value used to determine the expected Python type. This is
        mainly used when restoring nested dataclasses and collections.

    Returns
    -------
    Any
        Restored Python value.

    Raises
    ------
    KeyError
        If the requested value is not present.
    """
    if group.attrs.get(f"{name}__is_none", False):
        return None

    if name in group.attrs:
        value = group.attrs[name]

        if isinstance(value, bytes):
            return value.decode("utf-8")

        if isinstance(value, np.generic):
            return value.item()

        return value

    if name not in group:
        raise KeyError(f"{name!r} not found in HDF5 group {group.name!r}.")

    item = group[name]

    if isinstance(item, h5py.Dataset):
        value = item[()]

        if isinstance(value, bytes):
            value = value.decode("utf-8")

        elif isinstance(value, np.ndarray) and value.dtype.kind in {"S", "O", "U"}:
            value = [
                element.decode("utf-8") if isinstance(element, bytes) else element
                for element in value
            ]

        collection_type = item.attrs.get("collection_type")

        if isinstance(collection_type, bytes):
            collection_type = collection_type.decode("utf-8")

        if collection_type == "list":
            return value.tolist() if isinstance(value, np.ndarray) else list(value)

        if collection_type == "tuple":
            return tuple(value.tolist() if isinstance(value, np.ndarray) else value)

        if collection_type == "set":
            return set(value.tolist() if isinstance(value, np.ndarray) else value)

        return value

    if is_dataclass(current_value):
        for dataclass_field in fields(current_value):
            field_name = dataclass_field.name
            exists = (
                field_name in item
                or field_name in item.attrs
                or f"{field_name}__is_none" in item.attrs
            )

            if not exists:
                continue

            existing_field = getattr(current_value, field_name)
            restored_field = read_hdf5_value(item, field_name, existing_field)
            setattr(current_value, field_name, restored_field)

        return current_value

    collection_type = item.attrs.get("collection_type")

    if isinstance(collection_type, bytes):
        collection_type = collection_type.decode("utf-8")

    if collection_type in {"list", "tuple", "set"}:
        values = [
            read_hdf5_value(item, key)
            for key in sorted(item.keys(), key=int)
        ]

        if collection_type == "tuple":
            return tuple(values)

        if collection_type == "set":
            return set(values)

        return values

    result = {}

    names = set(item.keys())

    for attribute_name in item.attrs:
        if attribute_name.endswith("__is_none"):
            names.add(attribute_name.removesuffix("__is_none"))
        elif attribute_name != "collection_type":
            names.add(attribute_name)

    for child_name in names:
        result[child_name] = read_hdf5_value(item, child_name)

    return result


def save_full_hdf5(image_analysis, path: str | Path, guvs="all") -> None:
    """
    Save a complete GUVImage analysis into one HDF5 file.

    Parameters
    ----------
    image_analysis : GUVImage
        An initialized GUVImage containing the analysis results.

    path : str or pathlib.Path
        Destination HDF5 file.

    guvs : {"all", "good", "bad"} or iterable of int, default="all"
        GUVs to include. An explicit collection of GUV IDs may also be
        provided.

    Raises
    ------
    ValueError
        If an invalid string selection is supplied.

    KeyError
        If an explicitly selected GUV ID is not present.
    """
    path = Path(path)

    if isinstance(guvs, str):
        selection = guvs.lower()

        if selection == "all":
            guv_ids = list(image_analysis.guvs)
        elif selection == "good":
            guv_ids = list(image_analysis.good_GUVs)
        elif selection == "bad":
            guv_ids = list(image_analysis.bad_GUVs)
        else:
            raise ValueError("guvs must be 'all', 'good', 'bad', or a collection of IDs.")
    else:
        selection = "custom"
        guv_ids = list(guvs)

    missing_ids = [guv_id for guv_id in guv_ids if guv_id not in image_analysis.guvs]

    if missing_ids:
        raise KeyError(f"Unknown GUV IDs: {missing_ids}")

    guv_ids = sorted(set(guv_ids))
    selected_ids = set(guv_ids)

    with h5py.File(path, "w") as file:
        file.attrs["format_version"] = 1
        file.attrs["image_path"] = str(image_analysis.path)
        file.attrs["pixel_size"] = image_analysis.pixel_size
        file.attrs["selection"] = selection

        write_hdf5_value(file, "settings", image_analysis.settings)
        write_hdf5_value(file, "global_mask", image_analysis.global_mask)
        write_hdf5_value(file, "global_background", image_analysis.global_background)

        guvs_group = file.create_group("guvs")

        for guv_id in guv_ids:
            guv = image_analysis.guvs[guv_id]
            guv_group = guvs_group.create_group(str(guv_id))

            guv_group.attrs["id"] = guv.id
            guv_group.attrs["xc"] = guv.xc
            guv_group.attrs["yc"] = guv.yc
            guv_group.attrs["radius"] = guv.radius
            guv_group.attrs["death_mark"] = guv.death_mark
            guv_group.attrs["num_angles"] = guv.num_angles

            write_hdf5_value(guv_group, "ves_coordinates", guv.ves_coordinates)
            write_hdf5_value(guv_group, "analysis", guv.analysis)

        good_ids = np.asarray(sorted(selected_ids & image_analysis.good_GUVs), dtype=int)
        bad_ids = np.asarray(sorted(selected_ids & image_analysis.bad_GUVs), dtype=int)

        file.create_dataset("good_GUVs", data=good_ids)
        file.create_dataset("bad_GUVs", data=bad_ids)


def load_hdf5(hdf5_path: str | Path, path_override: str | Path | None = None):
    """
    Recreate a GUVImage from a saved HDF5 analysis file.

    The normal GUVImage constructor is first called using the saved image
    path and settings. This reloads the original image and detections.
    Saved GUV analysis results are then restored into the newly initialized
    objects.

    Parameters
    ----------
    hdf5_path : str or pathlib.Path
        HDF5 analysis file to load.

    path_override : str or pathlib.Path, optional
        Alternative path to the original image. This is useful when the
        image or project folder has moved since the HDF5 file was created.

    Returns
    -------
    GUVImage
        Reinitialized GUVImage containing the restored analysis results.

    Raises
    ------
    KeyError
        If a GUV saved in the HDF5 file is not present in the current
        detection file.
    """
    from .image import GUVImage

    hdf5_path = Path(hdf5_path)

    with h5py.File(hdf5_path, "r") as file:
        saved_path = file.attrs["image_path"]

        if isinstance(saved_path, bytes):
            saved_path = saved_path.decode("utf-8")

        image_path = str(path_override) if path_override is not None else str(saved_path)

        settings = AnalysisSettings()
        settings = read_hdf5_value(file, "settings", settings)

    image_analysis = GUVImage(path=image_path, settings=settings)

    with h5py.File(hdf5_path, "r") as file:
        saved_ids = {int(guv_id) for guv_id in file["guvs"].keys()}
        missing_ids = saved_ids - set(image_analysis.guvs)

        if missing_ids:
            raise KeyError(
                "Saved GUV IDs are not present in the current detection file: "
                f"{sorted(missing_ids)}"
            )

        image_analysis.guvs = {
            guv_id: image_analysis.guvs[guv_id]
            for guv_id in saved_ids
        }

        for guv_id in saved_ids:
            guv = image_analysis.guvs[guv_id]
            guv_group = file[f"guvs/{guv_id}"]

            guv.xc = float(guv_group.attrs["xc"])
            guv.yc = float(guv_group.attrs["yc"])
            guv.radius = float(guv_group.attrs["radius"])
            guv.num_angles = int(guv_group.attrs["num_angles"])
            guv.death_mark = bool(guv_group.attrs["death_mark"])

            guv.ves_coordinates = read_hdf5_value(guv_group, "ves_coordinates")
            guv.analysis = read_hdf5_value(guv_group, "analysis", guv.analysis)

        image_analysis.good_GUVs = set(file["good_GUVs"][()].astype(int))
        image_analysis.bad_GUVs = set(file["bad_GUVs"][()].astype(int))

        image_analysis.global_mask = read_hdf5_value(file, "global_mask")
        image_analysis.global_background = read_hdf5_value(file, "global_background")

        image_analysis.image_view._global_mask = image_analysis.global_mask
        image_analysis.image_view._global_background = image_analysis.global_background

    return image_analysis