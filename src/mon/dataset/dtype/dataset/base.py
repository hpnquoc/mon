#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements base classes for all datasets."""

from __future__ import annotations

__all__ = [
    "ChainDataset",
    "ConcatDataset",
    "Dataset",
    "IterableDataset",
    "VisionDataset",
    "Subset",
    "TensorDataset",
    "random_split",
]

from abc import ABC, abstractmethod
from typing import Any, Literal

import torch
from torch.utils.data import dataset
from torch.utils.data.dataset import *

from mon import core
from mon.core import Split, Task
from mon.dataset.dtype import annotation
from mon.dataset.dtype.transform import albumentation as A
from mon.globals import DEPTH_DATA_SOURCES

console             = core.console
ClassLabels         = core.ClassLabels
DatapointAttributes = annotation.DatapointAttributes
DepthMapAnnotation  = annotation.DepthMapAnnotation
ImageAnnotation     = annotation.ImageAnnotation


# region Base Dataset

class Dataset(dataset.Dataset, ABC):
    """Base class for all datasets.

    Attributes:
        tasks: List of supported tasks.
        splits: List of supported splits.
        has_test_annotations: If ``True``, test set has labels. Default is ``False``.
        datapoint_attrs: Dict of datapoint attributes (keys: names, values: types).
        classlabels: ``ClassLabels`` with supported labels. Default is ``None``.

    Args:
        root: Root dir with split subdirs. Default is ``None``.
        split: Data split to use. Default is ``Split.TRAIN``.
        transform: Transformations for input/target. Default is ``None``.
        to_tensor: If ``True``, converts to ``torch.Tensor``. Default is ``False``.
        cache_data: If ``True``, caches data to disk. Default is ``False``.
        verbose: If ``True``, enables verbose output. Default is ``False``.
    """
    
    tasks : list[Task]  = []
    splits: list[Split] = [Split.TRAIN, Split.VAL, Split.TEST, Split.PREDICT]
    datapoint_attrs     = DatapointAttributes({})
    has_test_annotations: bool        = False
    classlabels         : ClassLabels = None
    
    def __init__(
        self,
        root      : core.Path,
        split     : Split     = Split.TRAIN,
        transform : A.Compose = None,
        to_tensor : bool      = False,
        cache_data: bool      = False,
        verbose   : bool      = False,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.root       = core.Path(root)
        self.split      = split
        self.transform  = transform
        self.to_tensor  = to_tensor
        self.verbose    = verbose
        self.index      = 0  # Used with `__iter__` and `__next__`
        self.datapoints = {}
        self.init_transform()
        self.init_datapoints()
        self.init_data(cache_data=cache_data)
        
    # region Magic Methods
    
    def __del__(self):
        """Closes the dataset."""
        self.close()
    
    @abstractmethod
    def __getitem__(self, index: int) -> dict:
        """Gets a datapoint and metadata at given index.

        Args:
            index: Index of datapoint.

        Returns:
            Dict with datapoint and metadata.
        """
        pass
    
    def __iter__(self):
        """Gets total number of datapoints.

        Returns:
            Number of datapoints in dataset.
        """
        self.reset()
        return self
    
    @abstractmethod
    def __len__(self) -> int:
        """Gets the total number of datapoints.

        Returns:
            Number of datapoints in the dataset.
        """
        pass
    
    def __next__(self) -> dict:
        """Gets the next datapoint and metadata.

        Returns:
            Dict with next datapoint and metadata.

        Raises:
            StopIteration: If index exceeds dataset length.
        """
        if self.index >= self.__len__():
            raise StopIteration
        result = self.__getitem__(self.index)
        self.index += 1
        return result
    
    def __repr__(self) -> str:
        """Returns string representation of dataset.

        Returns:
            Formatted string with dataset details.
        """
        head = "Dataset " + self.__class__.__name__
        body = [f"Number of datapoints: {self.__len__()}"]
        if self.root:
            body.append(f"Root location: {self.root}")
        if hasattr(self, "transform") and self.transform:
            body += [repr(self.transform)]
        lines = [head]
        return "\n".join(lines)
    
    # endregion
    
    # region Properties
    
    @property
    def disable_pbar(self) -> bool:
        """Indicates if progress bar is disabled.

        Returns:
            ``True`` if progress bar disabled, ``False`` otherwise.
        """
        return not self.verbose
    
    @property
    def has_annotations(self) -> bool:
        """Checks if images have annotations.

        Returns:
            ``True`` if annotations exist, ``False`` otherwise.
        """
        return (
            (
                self.has_test_annotations
                and self.split in [Split.TEST, Split.PREDICT]
            )
            or (self.split in [Split.TRAIN, Split.VAL])
        )
    
    @property
    def hash(self) -> int:
        """Gets total hash value of all files.

        Returns:
            Integer sum of file hash values in bytes.
        """
        sum = 0
        for k, v in self.datapoints.items():
            if isinstance(v, list):
                for a in v:
                    if a and hasattr(a, "meta"):
                        sum += a.meta.get("hash", 0)
        return sum
    
    @property
    def main_attribute(self) -> str:
        """Gets the main dataset attribute.

        Returns:
            First key from ``datapoint_attrs`` as string.
        """
        return next(iter(self.datapoint_attrs.keys()))
    
    @property
    def new_datapoint(self) -> dict:
        """Creates a new datapoint with default values.

        Returns:
            Dict with attribute keys set to ``None``.
        """
        return {k: None for k in self.datapoint_attrs.keys()}
    
    @property
    def split(self) -> Split:
        """Gets the current dataset split.

        Returns:
            Current ``Split`` value.
        """
        return self._split
    
    @split.setter
    def split(self, split: Split):
        """Sets the dataset split.

        Args:
            split: Split value to set.

        Raises:
            ValueError: If ``split`` not in supported splits.
        """
        split = Split[split] if isinstance(split, str) else split
        if split in self.splits:
            self._split = split
        else:
            raise ValueError(f"[split] must be one of {self.splits}, got {split}.")
    
    @property
    def split_str(self) -> str:
        """Gets string representation of the split.

        Returns:
            String value of current split.
        """
        return self.split.value
    
    # endregion
    
    # region Initialization
    
    def init_transform(self, transform: A.Compose | dict = None):
        """Initializes transformation operations.

        Args:
            transform: Transformations to apply. Default is ``None``.
        """
        self.transform = transform or self.transform
    
    def init_datapoints(self):
        """Initializes the datapoints dictionary.

        Raises:
            ValueError: If ``datapoint_attrs`` has no attributes.
        """
        if not self.datapoint_attrs:
            raise ValueError("[datapoint_attrs] has no defined attributes")
        self.datapoints = {k: list[v]() for k, v in self.datapoint_attrs.items()}
    
    def init_data(self, cache_data: bool = False):
        """Initializes dataset data.

        Args:
            cache_data: If ``True``, caches data to disk. Default is ``False``.
        """
        cache_file = self.root / f"{self.split_str}.cache"
        if cache_data and cache_file.is_cache_file():
            self.load_cache(path=cache_file)
        else:
            self.get_data()
        
        self.filter_data()
        self.verify_data()
        
        if cache_data:
            self.cache_data(path=cache_file)
        else:
            core.delete_cache(cache_file)
    
    @abstractmethod
    def get_data(self):
        """Gets the base data."""
    
    def cache_data(self, path: core.Path):
        """Caches data to the specified path.

        Args:
            path: Path to save the cache.
        """
        hash_ = 0
        if path.is_cache_file():
            cache = torch.load(path)
            hash_ = cache.get("hash", 0)
        
        if self.hash != hash_:
            cache = self.datapoints | {"hash": self.hash}
            torch.save(cache, str(path))
            if self.verbose:
                console.log(f"Cached data to: {path}")
    
    def load_cache(self, path: core.Path):
        """Loads cached data from specified path.

        Args:
            path: Path to load cache from.
        """
        self.datapoints = torch.load(path)
        self.datapoints.pop("hash", None)
    
    @abstractmethod
    def filter_data(self):
        """Filters unwanted datapoints."""
    
    @abstractmethod
    def verify_data(self):
        """Verifies the dataset."""
    
    @abstractmethod
    def reset(self):
        """Resets the dataset."""
    
    @abstractmethod
    def close(self):
        """Closes and releases the dataset."""
    
    # endregion
    
    # region Retrieve Data
    
    @abstractmethod
    def get_datapoint(self, index: int) -> dict:
        """Gets a datapoint at specified index.

        Args:
            index: Index of datapoint.

        Returns:
            Dict containing the datapoint.
        """
    
    @abstractmethod
    def get_meta(self, index: int) -> dict:
        """Gets metadata at specified index.

        Args:
            index: Index of metadata.

        Returns:
            Dict containing the metadata.
        """
    
    @classmethod
    def collate_fn(cls, batch: list[dict]) -> dict:
        """Collates input items for batch processing.

        Args:
            batch: List of dicts from dataset.

        Returns:
            Collated dict for ``torch.utils.data.DataLoader``.
        """
        zipped = {
            k: list(v)
            for k, v in zip(batch[0].keys(), zip(*[b.values() for b in batch]))
        }
        for k, v in zipped.items():
            collate_fn = cls.datapoint_attrs.get_collate_fn(k)
            if collate_fn and v:
                zipped[k] = collate_fn(batch=v)
        return zipped
    
    # endregion

# endregion


# region Multimodal Dataset

class VisionDataset(Dataset, ABC):
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
    
    def init_transform(self, transform: A.Compose | Any = None):
        """Initializes transformations with multimodal support.

        Args:
            transform: Transformations to apply. Default is ``None``.
        """
        super().init_transform(transform=transform)
        if isinstance(self.transform, A.Compose):
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
            ref_images: list[ImageAnnotation] = []
            with core.get_progress_bar(disable=self.disable_pbar) as pbar:
                for img in pbar.track(
                    sequence    = images,
                    description = f"Listing {self.__class__.__name__} "
                                  f"{self.split_str} reference images"
                ):
                    root_name = img.root.name
                    path      = img.path.replace(f"/{root_name}/", f"/ref/")
                    ref_images.append(ImageAnnotation(
                        path = path.image_file(),
                        root = img.root
                    ))
            self.datapoints["ref_image"] = ref_images
    
    def get_depth_map(self):
        """Gets depth maps for the dataset."""
        images = self.datapoints.get("image", [])
        depths = self.datapoints.get("depth", [])
        
        if len(images) > 0 and len(depths) == 0:
            depths: list[DepthMapAnnotation] = []
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
                        DepthMapAnnotation(
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
            ref_depths: list[DepthMapAnnotation] = []
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
                        DepthMapAnnotation(
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
            console.log(f"Number of {self.split_str} datapoints: {self.__len__()}")
    
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

# endregion
