#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Flare7K Datasets."""

from __future__ import annotations

__all__ = [
    "Flare7KPPExtra",
    "Flare7KPPExtraDataModule",
    "Flare7KPPReal",
    "Flare7KPPRealDataModule",
    "Flare7KPPSynthetic",
    "Flare7KPPSyntheticDataModule",
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


@DATASETS.register(name="flare7k++_real")
class Flare7KPPReal(MultimodalDataset):
    """Flare7K++Real dataset consists of 100 flare/clear image pairs."""
    
    tasks : list[Task]  = [Task.NIGHTTIME]
    splits: list[Split] = [Split.TEST]
    datapoint_attrs     = DatapointAttributes({
        "image"    : ImageAnnotation,
        "ref_image": ImageAnnotation,
    })
    has_test_annotations: bool = True
    
    def __init__(self, root: core.Path = default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
    
    def get_data(self):
        patterns = [
            self.root / "flare7k++" / self.split_str / "real" / "image",
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
        

@DATASETS.register(name="flare7k++_synthetic")
class Flare7KPPSynthetic(MultimodalDataset):
    """Flare7K++Synthetic dataset consists of ``100`` flare/clear image pairs."""
    
    tasks : list[Task]  = [Task.NIGHTTIME]
    splits: list[Split] = [Split.TEST]
    datapoint_attrs     = DatapointAttributes({
        "image"    : ImageAnnotation,
        "ref_image": ImageAnnotation,
    })
    has_test_annotations: bool = True
    
    def __init__(self, root: core.Path = default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
    
    def get_data(self):
        patterns = [
            self.root / "flare7k++" / self.split_str / "synthetic" / "image",
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


@DATASETS.register(name="flare7k++_extra")
class Flare7KPPExtra(MultimodalDataset):
    """Flare7K++Extra dataset consists of ``100`` flare images."""
    
    tasks : list[Task]  = [Task.NIGHTTIME]
    splits: list[Split] = [Split.TEST]
    datapoint_attrs     = DatapointAttributes({
        "image"    : ImageAnnotation,
        # "ref_image": ImageAnnotation,
    })
    has_test_annotations: bool = False
    
    def __init__(self, root: core.Path = default_root_dir, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
    
    def get_data(self):
        patterns = [
            self.root / "flare7k++" / self.split_str / "extra" / "image",
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


@DATAMODULES.register(name="flare7k++_real")
class Flare7KPPRealDataModule(DataModule):
    
    tasks: list[Task] = [Task.NIGHTTIME]
    
    def prepare_data(self, *args, **kwargs):
        pass
    
    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
       
        if stage in [None, "train"]:
            self.train = Flare7KPPReal(split=Split.TEST, **self.dataset_kwargs)
            self.val   = Flare7KPPReal(split=Split.TEST, **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = Flare7KPPReal(split=Split.TEST, **self.dataset_kwargs)
        
        self.get_classlabels()
        if self.can_log:
            self.summarize()
    

@DATAMODULES.register(name="flare7k++_synthetic")
class Flare7KPPSyntheticDataModule(DataModule):
    
    tasks: list[Task] = [Task.NIGHTTIME]
    
    def prepare_data(self, *args, **kwargs):
        pass

    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        if stage in [None, "train"]:
            self.train = Flare7KPPExtra(split=Split.TEST, **self.dataset_kwargs)
            self.val   = Flare7KPPExtra(split=Split.TEST, **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = Flare7KPPExtra(split=Split.TEST, **self.dataset_kwargs)

        self.get_classlabels()
        if self.can_log:
            self.summarize()


@DATAMODULES.register(name="flare7k++_extra")
class Flare7KPPExtraDataModule(DataModule):
    
    tasks: list[Task] = [Task.NIGHTTIME]
    
    def prepare_data(self, *args, **kwargs):
        pass

    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")
        
        if stage in [None, "train"]:
            self.train = Flare7KPPExtra(split=Split.TEST, **self.dataset_kwargs)
            self.val   = Flare7KPPExtra(split=Split.TEST, **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = Flare7KPPExtra(split=Split.TEST, **self.dataset_kwargs)

        self.get_classlabels()
        if self.can_log:
            self.summarize()
