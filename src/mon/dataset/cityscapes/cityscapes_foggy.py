#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Cityscapes.

This module implements the Cityscapes dataset.

References:
	https://www.cityscapes-dataset.com/
"""

from __future__ import annotations

__all__ = [
    "CityscapesFoggy",
    "CityscapesFoggyDataModule",
]

from typing import Literal

import cv2

from mon import core
from mon.dataset import dtype
from mon.dataset.cityscapes.cityscapes import Cityscapes
from mon.globals import DATA_DIR, DATAMODULES, DATASETS, Split, Task

console                        = core.console
default_root_dir               = DATA_DIR / "cityscapes"
ClassLabels                    = dtype.ClassLabels
DataModule                     = dtype.DataModule
DatapointAttributes            = dtype.DatapointAttributes
DepthMapAnnotation             = dtype.DepthMapAnnotation
ImageAnnotation                = dtype.ImageAnnotation
MultimodalDataset              = dtype.MultimodalDataset
SemanticSegmentationAnnotation = dtype.SemanticSegmentationAnnotation


@DATASETS.register(name="cityscapes_foggy")
class CityscapesFoggy(Cityscapes):
    """Loads and processes the CityscapesFoggy dataset for dehazing tasks.

    Args:
        ``root``: Root directory path. Default is ``default_root_dir``.
    Raises:
        FileNotFoundError: If ``root``/cityscapes directory does not exist.
    """
    
    tasks : list[Task]  = [Task.DEHAZE]
    splits: list[Split] = [Split.TRAIN, Split.VAL, Split.TEST]
    datapoint_attrs     = DatapointAttributes({
        "image"    : ImageAnnotation,
        "ref_image": ImageAnnotation,
        "semantic" : SemanticSegmentationAnnotation,  # gtFine
    })
    has_test_annotations: bool = True
    
    def __init__(self, root: core.Path = default_root_dir, *args, **kwargs):
        root = root / "cityscapes" if root.name != "cityscapes" else root
        if not root.is_dir():
            raise FileNotFoundError(f"[root] must be a directory, but got [{root}]")
        super().__init__(root=root, *args, **kwargs)
    
    def get_data(self):
        """Loads foggy images, reference images, and semantic maps."""
        patterns = [self.root / self.split_str / "leftImg8bit_foggy"]
        
        images: list[ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                paths = sorted(pattern.rglob("*"))
                desc  = f"Listing {self.__class__.__name__} {self.split_str} images"
                for path in pbar.track(sequence=paths, description=desc):
                    if path.is_image_file():
                        images.append(ImageAnnotation(path=path, root=pattern))
        
        ref_images: list[ImageAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            desc = f"Listing {self.__class__.__name__} {self.split_str} reference images"
            for img in pbar.track(sequence=images, description=desc):
                path = img.path.replace("/leftImg8bit_foggy/", "/leftImg8bit/")
                stem = path.stem.split("leftImg8bit")[0]
                path = path.parent / f"{stem}leftImg8bit{path.suffix}"
                ref_images.append(ImageAnnotation(path=path.image_file(), root=img.root))
        
        # Semantic segmentation maps
        semantic: list[SemanticSegmentationAnnotation] = []
        with core.get_progress_bar(disable=self.disable_pbar) as pbar:
            desc = f"Listing {self.__class__.__name__} {self.split_str} semantic maps"
            for img in pbar.track(sequence=ref_images, description=desc):
                path = img.path.replace("/leftImg8bit/", "/gtFine/")
                semantic.append(SemanticSegmentationAnnotation(
                    path  = path.image_file(),
                    root  = img.root,
                    flags = cv2.IMREAD_GRAYSCALE
                ))
        
        self.datapoints["image"]     = images
        self.datapoints["ref_image"] = ref_images
        self.datapoints["semantic"]  = semantic


@DATAMODULES.register(name="cityscapes_foggy")
class CityscapesFoggyDataModule(DataModule):
    """Manages CityscapesFoggy dataset for training, validation, and testing.

    Args:
        ``stage``: Setup stage, one of "train", "test", "predict", or ``None``. Default is ``None``.
    """

    tasks: list[Task] = [Task.DERAIN]

    def prepare_data(self, *args, **kwargs):
        """Prepares data for the CityscapesFoggy dataset (currently a no-op)."""
        pass

    def setup(self, stage: Literal["train", "test", "predict", None] = None):
        """Sets up datasets for specified stage.

        Args:
            ``stage``: Stage to setup, one of "train", "test", "predict", or ``None``. Default is ``None``.
        """
        if self.can_log:
            console.log(f"Setup [red]{self.__class__.__name__}[/red].")

        if stage in [None, "train"]:
            self.train = CityscapesFoggy(split=Split.TRAIN, **self.dataset_kwargs)
            self.val   = CityscapesFoggy(split=Split.VAL,   **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = CityscapesFoggy(split=Split.TEST,  **self.dataset_kwargs)

        self.get_classlabels()
        if self.can_log:
            self.summarize()
