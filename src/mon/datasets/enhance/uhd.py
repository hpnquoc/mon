#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements UHD datasets."""

__all__ = [
    "UHD4K",
    "UHD4KDataModule",
    "UHD8K",
    "UHD8KDataModule",
]

from typing import Literal

from mon import core
from mon.datasets.core import *


# ----- Dataset -----
# @DATASETS.register(name="uhd4k")
class UHD4K(VisionDataset):
    """Loads UHD-4K dataset from ``root`` dir.

    Args:
        root: Directory path to dataset.
        *args: Additional args for parent class.
        **kwargs: Additional kwargs for parent class.

    Raises:
        FileNotFoundError: If ``root`` directory does not exist.
    """
    
    tasks : list[Task]  = [Task.LLE]
    splits: list[Split] = [Split.TRAIN, Split.TEST]
    datapoint_attrs     = DatapointAttributes({
        "image"    : Image,
        "depth"    : DepthMap,
        "ref_image": Image,
        "ref_depth": DepthMap,
    })
    has_test_annotations: bool = True

    def __init__(self, root: core.Path, *args, **kwargs):
        root = core.Path(root)
        root = root / "uhd" if root.name != "uhd" else root
        if not root.is_dir():
            raise FileNotFoundError(f"[root] directory not found: [{root}].")
        
        super().__init__(root=root, *args, **kwargs)

    def list_data(self):
        """Lists ``datapoints`` with image annotations for split."""
        patterns = [self.root / "4k" / self.split_str / "image"]

        images: list[Image] = []
        with core.create_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                paths = sorted(pattern.rglob("*"))
                desc  = f"Listing {self.__class__.__name__} {self.split_str} images"
                for path in pbar.track(sequence=paths, description=desc):
                    if path.is_image_file():
                        images.append(Image(path=path, root=pattern))

        self.datapoints["image"] = images


# @DATASETS.register(name="uhd8k")
class UHD8K(VisionDataset):
    """Loads UHD-8K dataset from ``root`` dir.

    Args:
        root: Directory path to dataset.
        *args: Additional args for parent class.
        **kwargs: Additional kwargs for parent class.

    Raises:
        FileNotFoundError: If ``root`` directory does not exist.
    """
    
    tasks : list[Task]  = [Task.LLE]
    splits: list[Split] = [Split.TRAIN, Split.TEST]
    datapoint_attrs     = DatapointAttributes({
        "image"    : Image,
        "depth"    : DepthMap,
        "ref_image": Image,
        "ref_depth": DepthMap,
    })
    has_test_annotations: bool = True

    def __init__(self, root: core.Path, *args, **kwargs):
        root = core.Path(root)
        root = root / "uhd" if root.name != "uhd" else root
        if not root.is_dir():
            raise FileNotFoundError(f"[root] directory not found: [{root}].")
        
        super().__init__(root=root, *args, **kwargs)

    def list_data(self):
        """Lists ``datapoints`` with image annotations for split."""
        patterns = [self.root / "8k" / self.split_str / "image"]

        images: list[Image] = []
        with core.create_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                paths = sorted(pattern.rglob("*"))
                desc  = f"Listing {self.__class__.__name__} {self.split_str} images"
                for path in pbar.track(sequence=paths, description=desc):
                    if path.is_image_file():
                        images.append(Image(path=path, root=pattern))

        self.datapoints["image"] = images

    
# ----- DataModule -----
# @DATAMODULES.register(name="uhd4k")
class UHD4KDataModule(core.DataModule):
    """Configures UHD-4K datasets for training/testing."""
    
    tasks: list[Task] = [Task.LLE]

    def prepare_data(self, *args, **kwargs):
        """Prepares data (placeholder, no action taken)."""
        pass

    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        """Sets up datasets for specified ``stage``.

        Args:
            stage: Stage to setup, one of ``"train"``, ``"test"``, ``"predict"``,
                or ``None``. Default is ``None``.
        """
        if self.can_log:
            core.console.log(f"Setup [red]{self.__class__.__name__}[/red].")

        if stage in [None, "train"]:
            self.train = UHD4K(split=Split.TRAIN, **self.dataset_kwargs)
            self.val   = UHD4K(split=Split.TEST,  **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = UHD4K(split=Split.TEST,  **self.dataset_kwargs)

        self.get_classes()
        if self.can_log:
            self.summarize()
            

# @DATAMODULES.register(name="uhd8k")
class UHD8KDataModule(core.DataModule):
    """Configures UHD-8K datasets for training/testing."""
    
    tasks: list[Task] = [Task.LLE]

    def prepare_data(self, *args, **kwargs):
        """Prepares data (placeholder, no action taken)."""
        pass

    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        """Sets up datasets for specified ``stage``.

        Args:
            stage: Stage to setup, one of ``"train"``, ``"test"``, ``"predict"``,
                or ``None``. Default is ``None``.
        """
        if self.can_log:
            core.console.log(f"Setup [red]{self.__class__.__name__}[/red].")

        if stage in [None, "train"]:
            self.train = UHD8K(split=Split.TRAIN, **self.dataset_kwargs)
            self.val   = UHD8K(split=Split.TEST,  **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = UHD8K(split=Split.TEST,  **self.dataset_kwargs)

        self.get_classes()
        if self.can_log:
            self.summarize()
