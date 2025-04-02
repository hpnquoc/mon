#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements Rain12 datasets."""

__all__ = [
    "Rain12",
    "Rain12DataModule",
]

from typing import Literal

from mon import core, vision
from mon.globals import DATA_DIR, DATAMODULES, DATASETS


@DATASETS.register(name="rain12")
class Rain12(vision.VisionDataset):
    """Loads Rain12 dataset from ``root`` dir.

    Args:
        root: Directory path to dataset. Default is ``DATA_DIR / "enhance"``.
        *args: Additional args for parent class.
        **kwargs: Additional kwargs for parent class.

    Raises:
        FileNotFoundError: If ``root`` directory does not exist.
    """

    tasks : list[core.Task]    = [core.Task.DERAIN]
    splits: list[core.Split]   = [core.Split.TRAIN]
    datapoint_attrs            = vision.DatapointAttributes({
        "image"    : vision.ImageAnnotation,
        "ref_image": vision.ImageAnnotation,
    })
    has_test_annotations: bool = False
    
    def __init__(self, root: core.Path = DATA_DIR / "enhance", *args, **kwargs):
        """Initializes dataset with ``root`` path and parent args."""
        root = root / "rain12" if root.name != "rain12" else root
        if not root.is_dir():
            raise FileNotFoundError(f"[root] directory not found: [{root}].")
        super().__init__(root=root, *args, **kwargs)
    
    def get_data(self):
        """Populates ``datapoints`` with image annotations for split."""
        patterns = [
            self.root / self.split_str / "image",
        ]
        
        images: list[vision.ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                paths = sorted(pattern.rglob("*"))
                desc  = f"Listing {self.__class__.__name__} {self.split_str} images"
                for path in pbar.track(sequence=paths, description=desc):
                    if path.is_image_file():
                        images.append(vision.ImageAnnotation(path=path, root=pattern))

        self.datapoints["image"] = images
        

@DATAMODULES.register(name="rain12")
class Rain12DataModule(core.DataModule):
    """Configures Rain12 datasets for training/testing.

    Args:
        *args: Additional args for parent class.
        **kwargs: Additional kwargs for parent class.
    """

    tasks: list[core.Task] = [core.Task.DERAIN]
    
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
            self.train = Rain12(split=core.Split.TRAIN, **self.dataset_kwargs)
            self.val   = Rain12(split=core.Split.TRAIN, **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = Rain12(split=core.Split.TRAIN, **self.dataset_kwargs)
        
        self.get_classlabels()
        if self.can_log:
            self.summarize()
