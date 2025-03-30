#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements base classes for all datasets."""

from __future__ import annotations

__all__ = [
    "DatapointAttributes",
    "VisionDataset",
]

from abc import ABC
from typing import Any, Literal, Optional

from mon import core
from mon.globals import DEPTH_DATA_SOURCES
from mon.vision.dtype import annotation as anno
from mon.vision.geometry import albumentation as album


class DatapointAttributes(dict[str: Optional[core.Annotation]]):
    """Holds datapoint attributes as a ``dict``.

    Args:
        args: Positional arguments for ``dict`` initialization.
        kwargs: Keyword arguments for ``dict`` initialization.

    Attributes:
        Keys: Attribute names as ``str``.
        Values: Annotation types as ``Annotation`` or ``None``.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def to_tensor_fns(self) -> dict[str: Optional[callable]]:
        """Returns dict of functions to convert annotation to tensor.
    
        Returns:
            Dict mapping keys to ``to_tensor`` functions or ``None``.
        """
        return {k: getattr(v, "to_tensor", None) for k, v in self.items() if v}
    
    def collate_fns(self) -> dict[str: Optional[callable]]:
        """Returns dict of functions to collate annotation.
    
        Returns:
            Dict mapping keys to ``collate_fn`` functions or ``None``.
        """
        return {k: getattr(v, "collate_fn", None) for k, v in self.items() if v}
    
    def get_tensor_fn(self, key: str) -> Optional[callable]:
        """Returns function to convert annotation to tensor.
    
        Args:
            key: Key of the annotation.
    
        Returns:
            ``to_tensor`` function or ``None`` if not found.
        """
        return self.to_tensor_fns().get(key, None)
    
    def get_collate_fn(self, key: str) -> Optional[callable]:
        """Returns function to collate annotation.
    
        Args:
            key: Key of the annotation.
    
        Returns:
            ``collate_fn`` function or ``None`` if not found.
        """
        return self.collate_fns().get(key, None)
    
    def get_albumentation_target_type(self, key: str) -> Optional[str]:
        """Returns Albumentations target type for an annotation.
    
        Args:
            key: Annotation object to check.
    
        Returns:
            Target type: ``"image"``, ``"mask"``, ``"bboxes"``, ``"keypoints"``, or
            ``"values"``; ``None`` if unknown.
        """
        v = self.get(key, None)
        if v in [anno.ImageAnnotation, anno.FrameAnnotation, anno.DepthMapAnnotation]:
            return "image"
        elif v in [anno.BBoxAnnotation, anno.BBoxesAnnotation]:
            return "bboxes"
        elif v in [core.ClassificationAnnotation, core.RegressionAnnotation]:
            return "values"
        elif v in [anno.SemanticSegmentationAnnotation]:
            return "mask"
        else:
            core.error_console.log(f"Unknown annotation type: {v}, {type(v)}.")
            return None
        # return self.albumentation_target_types().get(key, None)
        

class VisionDataset(core.Dataset, ABC):
    """Base class for multimodal, multi-task, multi-label datasets.

    Attributes:
        datapoint_attrs: Dict of attribute names and types. Common attributes:
            - ``'image'``    : ``ImageAnnotation`` (main attribute)
            - ``'depth'``    : ``DepthMapAnnotation``
            - ``'ref_image'``: ``ImageAnnotation``
            - ``'ref_depth'``: ``DepthMapAnnotation``

    Args:
        depth_source: Source of depth data. Default is ``'dav2_vitb'``.
    """
    
    def __init__(
        self,
        depth_source: Literal[*DEPTH_DATA_SOURCES] = "dav2_vitb",
        *args, **kwargs
    ):
        if depth_source not in DEPTH_DATA_SOURCES:
            raise ValueError(f"[depth_source] must be one of {DEPTH_DATA_SOURCES}, "
                             f"got {depth_source}.")
        self.depth_source = depth_source
        super().__init__(*args, **kwargs)
    
    # region Magic Methods
    
    def __getitem__(self, index: int) -> dict:
        """Gets a datapoint and metadata at specified index.

        Args:
            index: Index of datapoint.

        Returns:
            Dict with datapoint and metadata.
        """
        datapoint = self.get_datapoint(index=index)
        meta      = self.get_meta(index=index)
        
        if self.transform:
            main_attr      = self.main_attribute
            args           = {k: v for k, v in datapoint.items() if v is not None}
            args["image"]  = args.pop(main_attr)
            transformed    = self.transform(**args)
            transformed[main_attr] = transformed.pop("image")
            datapoint     |= transformed
        
        if self.to_tensor:
            for k, v in datapoint.items():
                to_tensor_fn = self.datapoint_attrs.get_tensor_fn(k)
                if to_tensor_fn and v is not None:
                    datapoint[k] = to_tensor_fn(v, normalize=True)
        
        return datapoint | {"meta": meta}
    
    def __len__(self) -> int:
        """Gets total number of datapoints.

        Returns:
            Number of datapoints in dataset.
        """
        return len(self.datapoints[self.main_attribute])
    
    # endregion
    
    # region Initialization
    
    def init_transform(self, transform: album.Compose | Any = None):
        """Initializes transformations with multimodal support.

        Args:
            transform: Transformations to apply. Default is ``None``.
        """
        super().init_transform(transform=transform)
        if isinstance(self.transform, album.Compose):
            additional_targets = self.datapoint_attrs.albumentation_target_types()
            additional_targets.pop(self.main_attribute, None)
            additional_targets.pop("meta", None)
            self.transform.add_targets(additional_targets)
    
    def init_data(self, cache_data: bool = False):
        """Initializes dataset data with multimodal support.

        Args:
            cache_data: If ``True``, caches data to disk. Default is ``False``.
        """
        cache_file = self.root / f"{self.split_str}.cache"
        if cache_data and cache_file.is_cache_file():
            self.load_cache(path=cache_file)
        else:
            self.get_data()
            self.get_multimodal_data()
        
        self.filter_data()
        self.verify_data()
        
        if cache_data:
            self.cache_data(path=cache_file)
        else:
            core.delete_cache(cache_file)
    
    def get_multimodal_data(self):
        """Gets multimodal data for the dataset."""
        if "depth" in self.datapoint_attrs:
            self.get_depth_map()
        
        if self.has_annotations:
            self.get_reference_image()
            if "ref_depth" in self.datapoint_attrs:
                self.get_reference_depth_map()
        else:
            self.datapoints.pop("ref_image", None)
            self.datapoints.pop("ref_depth", None)
    
    def get_reference_image(self):
        """Gets reference images for the dataset."""
        images     = self.datapoints.get("image",     [])
        ref_images = self.datapoints.get("ref_image", [])
        
        if len(ref_images) == 0:
            ref_images: list[anno.ImageAnnotation] = []
            with core.get_progress_bar(disable=self.disable_pbar) as pbar:
                for img in pbar.track(
                    sequence    = images,
                    description = f"Listing {self.__class__.__name__} "
                                  f"{self.split_str} reference images"
                ):
                    root_name = img.root.name
                    path      = img.path.replace(f"/{root_name}/", f"/ref/")
                    ref_images.append(anno.ImageAnnotation(
                        path = path.image_file(),
                        root = img.root
                    ))
            self.datapoints["ref_image"] = ref_images
    
    def get_depth_map(self):
        """Gets depth maps for the dataset."""
        images = self.datapoints.get("image", [])
        depths = self.datapoints.get("depth", [])
        
        if len(images) > 0 and len(depths) == 0:
            depths: list[anno.DepthMapAnnotation] = []
            with core.get_progress_bar(disable=self.disable_pbar) as pbar:
                for img in pbar.track(
                    sequence    = images,
                    description = f"Listing {self.__class__.__name__} "
                                  f"{self.split_str} depth maps"
                ):
                    root_name = img.root.name
                    path      = img.path.replace(f"/{root_name}/",
                                                 f"/{root_name}_{self.depth_source}/")
                    depths.append(
                        anno.DepthMapAnnotation(
                            path   = path.image_file(),
                            root   = img.root,
                            source = self.depth_source
                        )
                    )
            self.datapoints["depth"] = depths
            
    def get_reference_depth_map(self):
        """Gets reference depth maps for the dataset."""
        ref_images = self.datapoints.get("ref_image", [])
        ref_depths = self.datapoints.get("ref_depth", [])
        
        if len(ref_images) > 0 and len(ref_depths) == 0:
            ref_depths: list[anno.DepthMapAnnotation] = []
            with core.get_progress_bar(disable=self.disable_pbar) as pbar:
                for img in pbar.track(
                    sequence    = ref_images,
                    description = f"Listing {self.__class__.__name__} "
                                  f"{self.split_str} reference depth maps"
                ):
                    root_name = img.root.name
                    path      = img.path.replace(f"/{root_name}/",
                                                 f"/{root_name}_{self.depth_source}/")
                    ref_depths.append(
                        anno.DepthMapAnnotation(
                            path   = path.image_file(),
                            root   = img.root,
                            source = self.depth_source
                        )
                    )
            self.datapoints["ref_depth"] = ref_depths
    
    def filter_data(self):
        """Filter unwanted datapoints."""
        pass
    
    def verify_data(self):
        """Verifies dataset integrity.

        Raises:
            RuntimeError: If no datapoints or attributes invalid.
        """
        if self.__len__() <= 0:
            raise RuntimeError("No datapoints in the dataset")
        for k, v in self.datapoints.items():
            if k not in self.datapoint_attrs:
                raise RuntimeError(f"Attribute [{k}] is not defined in [datapoint_attrs]; "
                                   f"define it in the class if intentional.")
            if self.datapoint_attrs[k]:
                if v is None:
                    raise RuntimeError(f"No [{k}] attributes defined")
                if v is not None and len(v) != self.__len__():
                    raise RuntimeError(f"Number of [{k}] attributes ({len(v)}) does not "
                                       f"match datapoints ({self.__len__()}).")
        if self.verbose:
            core.console.log(f"Number of {self.split_str} datapoints: {self.__len__()}")
    
    def reset(self):
        """Resets the dataset to start over."""
        self.index = 0
    
    def close(self):
        """Closes and releases dataset resources."""
        pass
    
    # endregion
    
    # region Retrieve Data
    
    def get_datapoint(self, index: int) -> dict:
        """Gets a datapoint at specified index.

        Args:
            index: Index of datapoint.

        Returns:
            Dict containing datapoint data.
        """
        datapoint = self.new_datapoint
        for k, v in self.datapoints.items():
            if v is not None and v[index] and hasattr(v[index], "data"):
                datapoint[k] = v[index].data
        return datapoint
    
    def get_meta(self, index: int) -> dict:
        """Gets metadata at specified index.

        Args:
            index: Index of metadata.

        Returns:
            Dict with metadata from main attribute.
        """
        return self.datapoints[self.main_attribute][index].meta
    
    # endregion
