#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements GT-Snow datasets."""

__all__ = [
    "GTSnow",
    "GTSnowDataModule",
]

import os
from typing import Literal

from mon import core
from mon.datasets.core import *


# ----- Dataset -----
@DATASETS.register(name="gtsnow")
class GTSnow(VisionDataset):
    """Loads GTSnow dataset from ``root`` dir.

    Args:
        root: Directory path to dataset.
        *args: Additional args for parent class.
        **kwargs: Additional kwargs for parent class.

    Raises:
        FileNotFoundError: If ``root`` directory does not exist.
    """
    
    _tasks : list[Task]  = [Task.DESNOW]
    _splits: list[Split] = [Split.TRAIN]
    _datapoint_attrs     = DatapointAttributes({
        "image"    : Image,
        "ref_image": Image,
    })
    _has_test_annotations: bool = False

    def __init__(self, root: core.Path, *args, **kwargs):
        root = core.Path(root)
        root = root / "gtsnow" if root.name != "gtsnow" else root
        if not root.is_dir():
            raise FileNotFoundError(f"[root] directory not found: [{root}].")
        
        super().__init__(root=root, *args, **kwargs)

    def list_data(self):
        """Lists ``datapoints`` with image and ref annotations."""
        patterns = [self.root / self.split_str / "image"]

        images: list[Image] = []
        with core.create_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                paths = sorted(pattern.rglob("*"))
                desc  = f"Listing {self.__class__.__name__} {self.split_str} images"
                for path in pbar.track(sequence=paths, description=desc):
                    if path.is_image_file():
                        images.append(Image(path=path, root=pattern))

        ref_images: list[Image] = []
        with core.create_progress_bar(disable=self.disable_pbar) as pbar:
            desc = f"Listing {self.__class__.__name__} {self.split_str} reference images"
            for img in pbar.track(sequence=images, description=desc):
                path = str(img.path)
                path = path[:-9] + "C-000.png"
                path = path.replace(f"{os.sep}image{os.sep}", f"{os.sep}ref{os.sep}")
                path = core.Path(path)
                ref_images.append(Image(path=path.image_file(), root=img.root))

        self.datapoints["image"]     = images
        self.datapoints["ref_image"] = ref_images


# ----- DataModule -----
@DATAMODULES.register(name="gtsnow")
class GTSnowDataModule(core.DataModule):
    """Configures GTSnow datasets for training/testing."""
    
    _tasks: list[Task] = [Task.DESNOW]

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
            self.train = GTSnow(split=Split.TRAIN, **self.dataset_kwargs)
            self.val   = GTSnow(split=Split.TRAIN, **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = GTSnow(split=Split.TRAIN, **self.dataset_kwargs)

        if self.classlabels is None:
            self.get_classlabels()

        self.summarize()
