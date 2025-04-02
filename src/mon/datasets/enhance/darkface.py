#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements DarkFace datasets."""

__all__ = [
    "DarkFace",
    "DarkFaceDataModule",
    "DarkFaceFull",
    "DarkFaceFullDataModule",
]

from typing import Literal

from mon import core, vision
from mon.globals import DATA_DIR, DATAMODULES, DATASETS


@DATASETS.register(name="darkface")
class DarkFace(vision.VisionDataset):
    """Loads DarkFace dataset from ``root`` dir.

    Args:
        root: Directory path to dataset. Default is ``DATA_DIR / "enhance"``.
        *args: Additional args for parent class.
        **kwargs: Additional kwargs for parent class.

    Raises:
        FileNotFoundError: If ``root`` directory does not exist.
    """
    
    tasks : list[core.Task]    = [core.Task.LLIE, core.Task.DETECT]
    splits: list[core.Split]   = [core.Split.TEST]
    datapoint_attrs            = vision.DatapointAttributes({
        "image": vision.ImageAnnotation,
        "depth": vision.DepthMapAnnotation,
    })
    has_test_annotations: bool = False

    def __init__(self, root: core.Path = DATA_DIR / "enhance", *args, **kwargs):
        """Initializes dataset with ``root`` path and parent args."""
        root = root / "darkface" if root.name != "darkface" else root
        if not root.is_dir():
            raise FileNotFoundError(f"[root] directory not found: [{root}].")
        super().__init__(root=root, *args, **kwargs)

    def get_data(self):
        """Populates ``datapoints`` with image annotations for split."""
        patterns = [self.root / self.split_str / "image"]

        images: list[vision.ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                paths = sorted(pattern.rglob("*"))
                desc  = f"Listing {self.__class__.__name__} {self.split_str} images"
                for path in pbar.track(sequence=paths, description=desc):
                    if path.is_image_file():
                        images.append(vision.ImageAnnotation(path=path, root=pattern))

        self.datapoints["image"] = images
        

@DATASETS.register(name="darkface_full")
class DarkFaceFull(vision.VisionDataset):
    """Loads DarkFaceFull dataset from ``root`` dir.

    Args:
        root: Directory path to dataset. Default is ``DATA_DIR / "enhance"``.
        *args: Additional args for parent class.
        **kwargs: Additional kwargs for parent class.

    Raises:
        FileNotFoundError: If ``root`` directory does not exist.
    """
    
    tasks : list[core.Task]  = [core.Task.LLIE, core.Task.DETECT]
    splits: list[core.Split] = [core.Split.TEST]
    datapoint_attrs     = vision.DatapointAttributes({
        "image": vision.ImageAnnotation,
        "depth": vision.DepthMapAnnotation,
    })
    has_test_annotations: bool = False

    def __init__(self, root: core.Path = DATA_DIR / "enhance", *args, **kwargs):
        """Initializes dataset with ``root`` path and parent args."""
        root = root / "darkface" if root.name != "darkface" else root
        if not root.is_dir():
            raise FileNotFoundError(f"[root] directory not found: [{root}].")
        super().__init__(root=root, *args, **kwargs)

    def get_data(self):
        """Populates ``datapoints`` with image annotations for split."""
        patterns = [self.root / f"{self.split_str}_full" / "image"]

        images: list[vision.ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                paths = sorted(pattern.rglob("*"))
                desc  = f"Listing {self.__class__.__name__} {self.split_str} images"
                for path in pbar.track(sequence=paths, description=desc):
                    if path.is_image_file():
                        images.append(vision.ImageAnnotation(path=path, root=pattern))

        self.datapoints["image"] = images


@DATAMODULES.register(name="darkface")
class DarkFaceDataModule(core.DataModule):
    """Configures DarkFace datasets for training/testing.

    Args:
        *args: Additional args for parent class.
        **kwargs: Additional kwargs for parent class.
    """
    tasks: list[core.Task] = [core.Task.LLIE]

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
            self.train = DarkFace(split=core.Split.TEST, **self.dataset_kwargs)
            self.val   = DarkFace(split=core.Split.TEST, **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = DarkFace(split=core.Split.TEST, **self.dataset_kwargs)

        self.get_classlabels()
        if self.can_log:
            self.summarize()


@DATAMODULES.register(name="darkface_full")
class DarkFaceFullDataModule(core.DataModule):
    """Configures DarkFaceFull datasets for training/testing.

    Args:
        *args: Additional args for parent class.
        **kwargs: Additional kwargs for parent class.
    """
    tasks: list[core.Task] = [core.Task.LLIE]

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
            self.train = DarkFaceFull(split=core.Split.TEST, **self.dataset_kwargs)
            self.val   = DarkFaceFull(split=core.Split.TEST, **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = DarkFaceFull(split=core.Split.TEST, **self.dataset_kwargs)

        self.get_classlabels()
        if self.can_log:
            self.summarize()
