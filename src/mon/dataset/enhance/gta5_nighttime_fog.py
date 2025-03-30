#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements GTA5 Nighttime Fog datasets.

References:
    - https://github.com/jinyeying/nighttime_dehaze
"""

from __future__ import annotations

__all__ = [
    "GTA5NighttimeFog",
    "GTA5NighttimeFogDataModule",
]

from typing import Literal

from mon import core, vision
from mon.globals import DATA_DIR, DATAMODULES, DATASETS


@DATASETS.register(name="gta5_nighttime_fog")
class GTA5NighttimeFog(vision.VisionDataset):
    """Loads GTA5NighttimeFog dataset from ``root`` dir.

    Args:
        root: Directory path to dataset. Default is ``DATA_DIR / "enhance"``.
        *args: Additional args for parent class.
        **kwargs: Additional kwargs for parent class.

    Raises:
        FileNotFoundError: If ``root`` directory does not exist.
    """
    
    tasks : list[core.Task]    = [core.Task.DEHAZE, core.Task.NIGHTTIME]
    splits: list[core.Split]   = [core.Split.TRAIN, core.Split.TEST]
    datapoint_attrs            = vision.DatapointAttributes({
        "image"    : vision.ImageAnnotation,
        "depth"    : vision.DepthMapAnnotation,
        "ref_image": vision.ImageAnnotation,
        "ref_depth": vision.DepthMapAnnotation,
    })
    has_test_annotations: bool = True

    def __init__(self, root: core.Path = DATA_DIR / "enhance", *args, **kwargs):
        """Initializes dataset with ``root`` path and parent args."""
        root = root / "gta5_nighttime_fog" if root.name != "gta5_nighttime_fog" else root
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


@DATAMODULES.register(name="gta5_nighttime_fog")
class GTA5NighttimeFogDataModule(core.DataModule):
    """Configures GTA5NighttimeFog datasets for training/testing.

    Args:
        *args: Additional args for parent class.
        **kwargs: Additional kwargs for parent class.
    """
    
    tasks: list[core.Task] = [core.Task.DEHAZE, core.Task.NIGHTTIME]
    
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
            self.train = GTA5NighttimeFog(split=core.Split.TRAIN, **self.dataset_kwargs)
            self.val   = GTA5NighttimeFog(split=core.Split.TEST,  **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = GTA5NighttimeFog(split=core.Split.TEST,  **self.dataset_kwargs)
        
        self.get_classlabels()
        if self.can_log:
            self.summarize()
