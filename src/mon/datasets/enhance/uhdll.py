#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements UHD-LL datasets."""

__all__ = [
    "UHDLL",
    "UHDLLDataModule",
]

from typing import Literal

from mon.core import console, pathlib, rich, types
from mon.datasets.core import *


# ----- Dataset -----
@DATASETS.register(name="uhdll")
class UHDLL(VisionDataset):
    """Loads UHD-LL dataset from ``root`` dir.

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

    def __init__(self, root: pathlib.Path, *args, **kwargs):
        root = pathlib.Path(root)
        root = root / "uhdll" if root.name != "uhdll" else root
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
@DATAMODULES.register(name="uhdll")
class UHDLLDataModule(types.DataModule):
    """Configures UHD-LL datasets for training/testing."""
    
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
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")

        if stage in [None, "train"]:
            self.train = UHDLL(split=Split.TRAIN, **self.dataset_kwargs)
            self.val   = UHDLL(split=Split.TEST,  **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = UHDLL(split=Split.TEST,  **self.dataset_kwargs)

        self.get_classes()
        if self.can_log:
            self.summarize()
