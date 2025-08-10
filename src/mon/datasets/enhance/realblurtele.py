#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements RealBlur-Tele datasets."""

__all__ = [
    "RealBlurTeleJ",
    "RealBlurTeleJDataModule",
    "RealBlurTeleR",
    "RealBlurTeleRDataModule",
]

from typing import Literal

from mon.core import console, pathlib, rich, types
from mon.datasets.core import *


# ----- Dataset -----
@DATASETS.register(name="realblurtelej")
class RealBlurTeleJ(VisionDataset):
    """Loads RealBlurTele-J dataset from ``root`` dir.

    Args:
        root: Directory path to dataset.
        *args: Additional args for parent class.
        **kwargs: Additional kwargs for parent class.

    Raises:
        FileNotFoundError: If ``root`` directory does not exist.
    """
    
    tasks : list[Task]  = [Task.DEBLUR, Task.LLE]
    splits: list[Split] = [Split.TEST]
    datapoint_attrs     = DatapointAttributes({
        "image"    : Image,
        "ref_image": Image,
    })
    has_test_annotations: bool = False

    def __init__(self, root: pathlib.Path, *args, **kwargs):
        root = pathlib.Path(root)
        root = root / "realblurtele" if root.name != "realblurtele" else root
        if not root.is_dir():
            raise FileNotFoundError(f"[root] directory not found: [{root}].")
        
        super().__init__(root=root, *args, **kwargs)

    def list_data(self):
        """Lists ``datapoints`` with image annotations for split."""
        patterns = [self.root / self.split_str / "j" / "image"]

        images: list[Image] = []
        with rich.create_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                paths = sorted(pattern.rglob("*"))
                desc  = f"Listing {self.__class__.__name__} {self.split_str} images"
                for path in pbar.track(sequence=paths, description=desc):
                    if path.is_image_file():
                        images.append(Image(path=path, root=pattern))

        self.datapoints["image"] = images


@DATASETS.register(name="realblurteler")
class RealBlurTeleR(VisionDataset):
    """Loads RealBlurTele-R dataset from ``root`` dir.

    Args:
        root: Directory path to dataset.
        *args: Additional args for parent class.
        **kwargs: Additional kwargs for parent class.

    Raises:
        FileNotFoundError: If ``root`` directory does not exist.
    """

    tasks : list[Task]  = [Task.DEBLUR, Task.LLE]
    splits: list[Split] = [Split.TEST]
    datapoint_attrs     = DatapointAttributes({
        "image"    : Image,
        "ref_image": Image,
    })
    has_test_annotations: bool = False

    def __init__(self, root: pathlib.Path, *args, **kwargs):
        root = pathlib.Path(root)
        root = root / "realblurtele" if root.name != "realblurtele" else root
        if not root.is_dir():
            raise FileNotFoundError(f"[root] directory not found: [{root}].")

        super().__init__(root=root, *args, **kwargs)

    def list_data(self):
        """Lists ``datapoints`` with image annotations for split."""
        patterns = [self.root / self.split_str / "r" / "image"]

        images: list[Image] = []
        with rich.create_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                paths = sorted(pattern.rglob("*"))
                desc  = f"Listing {self.__class__.__name__} {self.split_str} images"
                for path in pbar.track(sequence=paths, description=desc):
                    if path.is_image_file():
                        images.append(Image(path=path))

        self.datapoints["image"] = images


# ----- DataModule -----
@DATAMODULES.register(name="realblurtelej")
class RealBlurTeleJDataModule(types.DataModule):
    """Configures RealBlurTele-J datasets for training/testing."""
    
    tasks: list[Task] = [Task.DEBLUR, Task.LLE]

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
            self.train = RealBlurTeleJ(split=Split.TRAIN, **self.dataset_kwargs)
            self.val   = RealBlurTeleJ(split=Split.TEST, **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = RealBlurTeleJ(split=Split.TEST, **self.dataset_kwargs)

        self.get_classes()
        if self.can_log:
            self.summarize()


@DATAMODULES.register(name="realblurteler")
class RealBlurTeleRDataModule(types.DataModule):
    """Configures RealBlurTele-R datasets for training/testing."""

    tasks: list[Task] = [Task.DEBLUR, Task.LLE]

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
            self.train = RealBlurTeleR(split=Split.TRAIN, **self.dataset_kwargs)
            self.val   = RealBlurTeleR(split=Split.TEST, **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = RealBlurTeleR(split=Split.TEST, **self.dataset_kwargs)

        self.get_classes()
        if self.can_log:
            self.summarize()
