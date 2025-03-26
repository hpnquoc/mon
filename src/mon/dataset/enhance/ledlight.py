#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""LEDLight Datasets."""

from __future__ import annotations

__all__ = [
    "LEDLight",
    "LEDLightDataModule",
]

from typing import Literal

from mon import core
from mon.dataset import dtype
from mon.globals import DATA_DIR, DATAMODULES, DATASETS, Split, Task

console             = core.console
default_root_dir    = DATA_DIR / "enhance"
ClassLabels         = dtype.ClassLabels
DataModule          = dtype.DataModule
DatapointAttributes = dtype.DatapointAttributes
DepthMapAnnotation  = dtype.DepthMapAnnotation
ImageAnnotation     = dtype.ImageAnnotation
MultimodalDataset   = dtype.MultimodalDataset


@DATASETS.register(name="ledlight")
class LEDLight(MultimodalDataset):
    """Loads LEDLight dataset from ``root`` dir.

    Args:
        root: Directory path to dataset. Default is ``default_root_dir``
        *args: Additional args for parent class.
        **kwargs: Additional kwargs for parent class.

    Raises:
        FileNotFoundError: If ``root`` directory does not exist.
    """
    tasks : list[Task]  = [Task.NIGHTTIME]
    splits: list[Split] = [Split.TEST]
    datapoint_attrs     = DatapointAttributes({
        "image"    : ImageAnnotation,
        "ref_image": ImageAnnotation,
    })
    has_test_annotations: bool = True

    def __init__(self, root: core.Path = default_root_dir, *args, **kwargs):
        """Initializes dataset with ``root`` path and parent args."""
        root = root / "ledlight" if root.name != "ledlight" else root
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
        

@DATAMODULES.register(name="ledlight")
class LEDLightDataModule(DataModule):
    """Configures LEDLight datasets for training/testing.

    Args:
        *args: Additional args for parent class.
        **kwargs: Additional kwargs for parent class.
    """
    
    tasks: list[Task] = [Task.NIGHTTIME]

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
            self.train = LEDLight(split=Split.TEST, **self.dataset_kwargs)
            self.val   = LEDLight(split=Split.TEST, **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = LEDLight(split=Split.TEST, **self.dataset_kwargs)

        self.get_classlabels()
        if self.can_log:
            self.summarize()
