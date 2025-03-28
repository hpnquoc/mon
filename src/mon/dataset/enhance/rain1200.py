#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements Rain1200 datasets."""

from __future__ import annotations

__all__ = [
    "Rain1200",
    "Rain1200DataModule",
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


@DATASETS.register(name="rain1200")
class Rain1200(MultimodalDataset):
    """Loads Rain1200 dataset from ``root`` dir.

    Args:
        root: Directory path to dataset. Default is ``default_root_dir``.
        *args: Additional args for parent class.
        **kwargs: Additional kwargs for parent class.

    Raises:
        FileNotFoundError: If ``root`` directory does not exist.
    """

    tasks : list[Task]  = [Task.DERAIN]
    splits: list[Split] = [Split.TRAIN, Split.VAL, Split.TEST]
    datapoint_attrs     = DatapointAttributes({
        "image"    : ImageAnnotation,
        "ref_image": ImageAnnotation,
    })
    has_test_annotations: bool = True
    
    def __init__(self, root: core.Path = default_root_dir, *args, **kwargs):
        """Initializes dataset with ``root`` path and parent args."""
        root = root / "rain1200" if root.name != "rain1200" else root
        if not root.is_dir():
            raise FileNotFoundError(f"[root] directory not found: [{root}].")
        super().__init__(root=root, *args, **kwargs)
    
    def get_data(self):
        """Populates ``datapoints`` with image annotations for split."""
        if self.split in [Split.TRAIN]:
            patterns = [
                self.root / self.split_str / "light"  / "image",
                self.root / self.split_str / "medium" / "image",
                self.root / self.split_str / "heavy"  / "image",
            ]
        else:
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


@DATAMODULES.register(name="rain1200")
class Rain1200DataModule(DataModule):
    """Configures Rain1200 datasets for training/testing.

    Args:
        *args: Additional args for parent class.
        **kwargs: Additional kwargs for parent class.
    """

    tasks: list[Task] = [Task.DERAIN]
    
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
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        if stage in [None, "train"]:
            self.train = Rain1200(split=Split.TRAIN, **self.dataset_kwargs)
            self.val   = Rain1200(split=Split.VAL,   **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = Rain1200(split=Split.TEST,  **self.dataset_kwargs)
        
        self.get_classlabels()
        if self.can_log:
            self.summarize()
