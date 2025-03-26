#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Rain100 Datasets."""

from __future__ import annotations

__all__ = [
    "Rain100",
    "Rain100DataModule",
    "Rain100H",
    "Rain100HDataModule",
    "Rain100L",
    "Rain100LDataModule",
]

from typing import Literal

from mon import core
from mon.dataset import dtype
from mon.globals import DATA_DIR, DATAMODULES, DATASETS, Split, Task

console             = core.console
default_root_dir    = DATA_DIR / "enhance"
DataModule          = dtype.DataModule
DatapointAttributes = dtype.DatapointAttributes
DepthMapAnnotation  = dtype.DepthMapAnnotation
ImageAnnotation     = dtype.ImageAnnotation
MultimodalDataset   = dtype.MultimodalDataset


@DATASETS.register(name="rain100")
class Rain100(MultimodalDataset):
    """Loads Rain100 dataset from ``root`` dir.

    Args:
        root: Directory path to dataset. Default is ``default_root_dir``.
        *args: Additional args for parent class.
        **kwargs: Additional kwargs for parent class.

    Raises:
        FileNotFoundError: If ``root`` directory does not exist.
    """

    tasks : list[Task]  = [Task.DERAIN]
    splits: list[Split] = [Split.TEST]
    datapoint_attrs     = DatapointAttributes({
        "image"    : ImageAnnotation,
        "ref_image": ImageAnnotation,
    })
    has_test_annotations: bool = True
    
    def __init__(self, root: core.Path = default_root_dir, *args, **kwargs):
        """Initializes dataset with ``root`` path and parent args."""
        root = root / "rain100" if root.name != "rain100" else root
        if not root.is_dir():
            raise FileNotFoundError(f"[root] directory not found: [{root}]")
        super().__init__(root=root, *args, **kwargs)
    
    def get_data(self):
        """Populates ``datapoints`` with image annotations for split."""
        patterns = [
            self.root / self.split_str / "image",
        ]
        
        images: list[ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                paths = sorted(pattern.rglob("*"))
                desc  = f"Listing {self.__class__.__name__} {self.split_str} images"
                for path in pbar.track(sequence=paths, description=desc):
                    if path.is_image_file():
                        images.append(ImageAnnotation(path=path, root=pattern))
        
        self.datapoints["image"] = images


@DATASETS.register(name="rain100h")
class Rain100H(MultimodalDataset):
    """Loads Rain100H dataset from ``root`` dir.

    Args:
        root: Directory path to dataset. Default is ``default_root_dir``.
        *args: Additional args for parent class.
        **kwargs: Additional kwargs for parent class.

    Raises:
        FileNotFoundError: If ``root`` directory does not exist.
    """

    tasks : list[Task]  = [Task.DERAIN]
    splits: list[Split] = [Split.TRAIN, Split.TEST]
    datapoint_attrs     = DatapointAttributes({
        "image"    : ImageAnnotation,
        "ref_image": ImageAnnotation,
    })
    has_test_annotations: bool = True
    
    def __init__(self, root: core.Path = default_root_dir, *args, **kwargs):
        """Initializes dataset with ``root`` path and parent args."""
        root = root / "rain100h" if root.name != "rain100h" else root
        if not root.is_dir():
            raise FileNotFoundError(f"[root] directory not found: [{root}]")
        super().__init__(root=root, *args, **kwargs)
    
    def get_data(self):
        """Populates ``datapoints`` with image annotations for split."""
        patterns = [
            self.root / self.split_str / "image",
        ]
        
        images: list[ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                paths = sorted(pattern.rglob("*"))
                desc  = f"Listing {self.__class__.__name__} {self.split_str} images"
                for path in pbar.track(sequence=paths, description=desc):
                    if path.is_image_file():
                        images.append(ImageAnnotation(path=path, root=pattern))
        
        self.datapoints["image"] = images
        

@DATASETS.register(name="rain100l")
class Rain100L(MultimodalDataset):
    """Loads Rain100L dataset from ``root`` dir.

    Args:
        root: Directory path to dataset. Default is ``default_root_dir``.
        *args: Additional args for parent class.
        **kwargs: Additional kwargs for parent class.

    Raises:
        FileNotFoundError: If ``root`` directory does not exist.
    """

    tasks : list[Task]  = [Task.DERAIN]
    splits: list[Split] = [Split.TRAIN, Split.TEST]
    datapoint_attrs     = DatapointAttributes({
        "image"    : ImageAnnotation,
        "ref_image": ImageAnnotation,
    })
    has_test_annotations: bool = True
    
    def __init__(self, root: core.Path = default_root_dir, *args, **kwargs):
        """Initializes dataset with ``root`` path and parent args."""
        root = root / "rain100l" if root.name != "rain100l" else root
        if not root.is_dir():
            raise FileNotFoundError(f"[root] directory not found: [{root}]")
        super().__init__(root=root, *args, **kwargs)

    def get_data(self):
        """Populates ``datapoints`` with image annotations for split."""
        patterns = [
            self.root / "rain100l" / self.split_str / "image"
        ]
        
        images: list[ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                paths = sorted(pattern.rglob("*"))
                desc  = f"Listing {self.__class__.__name__} {self.split_str} images"
                for path in pbar.track(sequence=paths, description=desc):
                    if path.is_image_file():
                        images.append(ImageAnnotation(path=path, root=pattern))
        
        self.datapoints["image"] = images
        
   
@DATAMODULES.register(name="rain100")
class Rain100DataModule(DataModule):
    """Configures Rain100 datasets for training/testing.

    Args:
        *args: Additional args for parent class.
        **kwargs: Additional kwargs for parent class.
    """

    tasks: list[Task] = [Task.DERAIN]
    
    def prepare_data(self, *args, **kwargs):
        """Prepares data (placeholder, no action taken)."""
        pass
    
    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        """Sets up datasets for given ``stage``.

        Args:
            stage: Stage to configure. Default is ``None``.
        """
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        if stage in [None, "train"]:
            self.train = Rain100(split=Split.TEST, **self.dataset_kwargs)
            self.val   = Rain100(split=Split.TEST, **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = Rain100(split=Split.TEST, **self.dataset_kwargs)
        
        self.get_classlabels()
        if self.can_log:
            self.summarize()


@DATAMODULES.register(name="rain100h")
class Rain100HDataModule(DataModule):
    """Configures Rain100H datasets for training/testing.

    Args:
        *args: Additional args for parent class.
        **kwargs: Additional kwargs for parent class.
    """

    tasks: list[Task] = [Task.DERAIN]
    
    def prepare_data(self, *args, **kwargs):
        """Prepares data (placeholder, no action taken)."""
        pass
    
    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        """Sets up datasets for given ``stage``.

        Args:
            stage: Stage to configure. Default is ``None``.
        """
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        if stage in [None, "train"]:
            self.train = Rain100H(split=Split.TRAIN, **self.dataset_kwargs)
            self.val   = Rain100H(split=Split.TEST,  **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = Rain100H(split=Split.TEST,  **self.dataset_kwargs)
        
        self.get_classlabels()
        if self.can_log:
            self.summarize()


@DATAMODULES.register(name="rain100l")
class Rain100LDataModule(DataModule):
    """Configures Rain100L datasets for training/testing.

    Args:
        *args: Additional args for parent class.
        **kwargs: Additional kwargs for parent class.
    """

    tasks: list[Task] = [Task.DERAIN]
    
    def prepare_data(self, *args, **kwargs):
        """Prepares data (placeholder, no action taken)."""
        pass
    
    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        """Sets up datasets for given ``stage``.

        Args:
            stage: Stage to configure. Default is ``None``.
        """
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        if stage in [None, "train"]:
            self.train = Rain100L(split=Split.TRAIN, **self.dataset_kwargs)
            self.val   = Rain100L(split=Split.TRAIN, **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = Rain100L(split=Split.TEST,  **self.dataset_kwargs)
        
        self.get_classlabels()
        if self.can_log:
            self.summarize()
