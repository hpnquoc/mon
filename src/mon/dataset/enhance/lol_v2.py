#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""LOL-v2 Datasets."""

from __future__ import annotations

__all__ = [
    "LOLv2Real",
    "LOLv2RealDataModule",
    "LOLv2Synthetic",
    "LOLv2SyntheticDataModule",
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


@DATASETS.register(name="lol_v2_real")
class LOLv2Real(MultimodalDataset):
    """Loads LOL-v2 Real dataset from ``root`` dir.

    Args:
        root: Directory path to dataset. Default is ``default_root_dir``.
        *args: Additional args for parent class.
        **kwargs: Additional kwargs for parent class.

    Raises:
        FileNotFoundError: If ``root`` directory does not exist.
    """
    
    tasks : list[Task]  = [Task.LLIE]
    splits: list[Split] = [Split.TRAIN, Split.TEST]
    datapoint_attrs     = DatapointAttributes({
        "image"    : ImageAnnotation,
        "depth"    : DepthMapAnnotation,
        "ref_image": ImageAnnotation,
        "ref_depth": DepthMapAnnotation,
    })
    has_test_annotations: bool = True

    def __init__(self, root: core.Path = default_root_dir, *args, **kwargs):
        """Initializes dataset with ``root`` path and parent args."""
        root = root / "lol_v2_real" if root.name != "lol_v2_real" else root
        if not root.is_dir():
            raise FileNotFoundError(f"[root] directory not found: [{root}]")
        super().__init__(root=root, *args, **kwargs)

    def get_data(self):
        """Populates ``datapoints`` with image annotations for split."""
        patterns = [self.root / self.split_str / "image"]

        images: list[ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                paths = sorted(pattern.rglob("*"))
                desc  = f"Listing {self.__class__.__name__} {self.split_str} images"
                for path in pbar.track(sequence=paths, description=desc):
                    if path.is_image_file():
                        images.append(ImageAnnotation(path=path, root=pattern))

        self.datapoints["image"] = images


@DATASETS.register(name="lol_v2_synthetic")
class LOLv2Synthetic(MultimodalDataset):
    """Loads LOL-v2 Synthetic dataset from ``root`` dir.

    Args:
        root: Directory path to dataset. Default is ``default_root_dir``.
        *args: Additional args for parent class.
        **kwargs: Additional kwargs for parent class.

    Raises:
        FileNotFoundError: If ``root`` directory does not exist.
    """
    
    tasks : list[Task]  = [Task.LLIE]
    splits: list[Split] = [Split.TRAIN, Split.TEST]
    datapoint_attrs     = DatapointAttributes({
        "image"    : ImageAnnotation,
        "depth"    : DepthMapAnnotation,
        "ref_image": ImageAnnotation,
        "ref_depth": DepthMapAnnotation,
    })
    has_test_annotations: bool = True

    def __init__(self, root: core.Path = default_root_dir, *args, **kwargs):
        """Initializes dataset with ``root`` path and parent args."""
        root = root / "lol_v2_synthetic" if root.name != "lol_v2_synthetic" else root
        if not root.is_dir():
            raise FileNotFoundError(f"[root] directory not found: [{root}]")
        super().__init__(root=root, *args, **kwargs)

    def get_data(self):
        """Populates ``datapoints`` with image annotations for split."""
        patterns = [self.root / self.split_str / "image"]

        images: list[ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                paths = sorted(pattern.rglob("*"))
                desc  = f"Listing {self.__class__.__name__} {self.split_str} images"
                for path in pbar.track(sequence=paths, description=desc):
                    if path.is_image_file():
                        images.append(ImageAnnotation(path=path, root=pattern))

        self.datapoints["image"] = images

    
@DATAMODULES.register(name="lol_v2_real")
class LOLv2RealDataModule(DataModule):
    """Configures LOLv2Real datasets for training/testing.

    Args:
        *args: Additional args for parent class.
        **kwargs: Additional kwargs for parent class.
    """
    
    tasks: list[Task] = [Task.LLIE]

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
            self.train = LOLv2Real(split=Split.TRAIN, **self.dataset_kwargs)
            self.val   = LOLv2Real(split=Split.TEST,  **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = LOLv2Real(split=Split.TEST,  **self.dataset_kwargs)

        self.get_classlabels()
        if self.can_log:
            self.summarize()
            

@DATAMODULES.register(name="lol_v2_synthetic")
class LOLv2SyntheticDataModule(DataModule):
    """Configures LOLv2Synthetic datasets for training/testing.

    Args:
        *args: Additional args for parent class.
        **kwargs: Additional kwargs for parent class.
    """
    
    tasks: list[Task] = [Task.LLIE]

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
            self.train = LOLv2Synthetic(split=Split.TRAIN, **self.dataset_kwargs)
            self.val   = LOLv2Synthetic(split=Split.TEST,  **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = LOLv2Synthetic(split=Split.TEST,  **self.dataset_kwargs)

        self.get_classlabels()
        if self.can_log:
            self.summarize()
