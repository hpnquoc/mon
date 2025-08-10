#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements Dense-NH-Haze datasets."""

__all__ = [
    "DenseNHHaze",
    "DenseNHHazeDataModule",
]

from typing import Literal

from mon.core import console, pathlib, rich, types
from mon.datasets.core import *


# ----- Dataset -----
@DATASETS.register(name="densenhhaze")
class DenseNHHaze(VisionDataset):
    """Loads Dense-NH-Haze dataset from ``root`` dir.

    Args:
        root: Directory path to dataset.
        *args: Additional args for parent class.
        **kwargs: Additional kwargs for parent class.

    Raises:
        FileNotFoundError: If ``root`` directory does not exist.
    """
    
    tasks : list[Task]  = [Task.DEHAZE]
    splits: list[Split] = [Split.TRAIN, Split.VAL, Split.TEST]
    datapoint_attrs     = DatapointAttributes({
        "image"    : Image,
        "depth"    : DepthMap,
        "ref_image": Image,
        "ref_depth": DepthMap,
    })
    has_test_annotations: bool = False

    def __init__(self, root: pathlib.Path, *args, **kwargs):
        root = pathlib.Path(root)
        root = root / "densenhhaze" if root.name != "densenhhaze" else root
        if not root.is_dir():
            raise FileNotFoundError(f"[root] directory not found: [{root}].")

        super().__init__(root=root, *args, **kwargs)

    def list_data(self):
        """Lists ``datapoints`` with image annotations for split."""
        patterns = [self.root / self.split_str / "image"]

        images: list[Image] = []
        with rich.create_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                paths = sorted(pattern.rglob("*"))
                desc  = f"Listing {self.__class__.__name__} {self.split_str} images"
                for path in pbar.track(sequence=paths, description=desc):
                    if path.is_image_file():
                        images.append(Image(path=path, root=pattern))

        self.datapoints["image"] = images


# ----- DataModule -----
@DATAMODULES.register(name="densenhhaze")
class DenseNHHazeDataModule(types.DataModule):
    """Configures DenseHaze datasets for training/testing."""
    
    tasks: list[Task] = [Task.DEHAZE]

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
            self.train = DenseNHHaze(split=Split.TRAIN, **self.dataset_kwargs)
            self.val   = DenseNHHaze(split=Split.VAL,   **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = DenseNHHaze(split=Split.TEST,  **self.dataset_kwargs)

        self.get_classes()
        if self.can_log:
            self.summarize()
