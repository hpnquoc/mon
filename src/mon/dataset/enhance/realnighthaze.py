#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""RealNightHaze Datasets."""

from __future__ import annotations

__all__ = [
    "RealNightHaze",
    "RealNightHazeDataModule",
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


# region Dataset

@DATASETS.register(name="realnighthaze")
class RealNightHaze(MultimodalDataset):
    
    tasks : list[Task]  = [Task.NIGHTTIME]
    splits: list[Split] = [Split.TEST]
    datapoint_attrs     = DatapointAttributes({
        "image": ImageAnnotation,
        "depth": DepthMapAnnotation,
    })
    has_test_annotations: bool = False
    
    def __init__(self, root: core.Path = default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
    
    def get_data(self):
        patterns = [
            self.root / "realnighthaze" / self.split_str / "image",
        ]
        
        # Images
        images: list[ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                for path in pbar.track(
                    sequence    = sorted(list(pattern.rglob("*"))),
                    description = f"Listing {self.__class__.__name__} {self.split_str} images"
                ):
                    if path.is_image_file():
                        images.append(ImageAnnotation(path=path, root=pattern))
          
        self.datapoints["image"] = images


@DATAMODULES.register(name="realnighthaze")
class RealNightHazeDataModule(DataModule):
    
    tasks: list[Task] = [Task.NIGHTTIME]
    
    def prepare_data(self, *args, **kwargs):
        pass
    
    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        if stage in [None, "train"]:
            self.train = RealNightHaze(split=Split.TEST, **self.dataset_kwargs)
            self.val   = RealNightHaze(split=Split.TEST, **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = RealNightHaze(split=Split.TEST, **self.dataset_kwargs)
        
        self.get_classlabels()
        if self.can_log:
            self.summarize()
