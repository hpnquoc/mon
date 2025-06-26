#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements the Cityscapes Foggy dataset.

References:
	- https://www.cityscapes-dataset.com/
"""

__all__ = [
    "CityscapesFoggy",
    "CityscapesFoggyDataModule",
]

import os
from typing import Literal

import cv2

from mon import core
from mon.datasets.cityscapes.cityscapes import Cityscapes
from mon.datasets.core import *


# ----- Dataset -----
@DATASETS.register(name="cityscapes_foggy")
class CityscapesFoggy(Cityscapes):
    """Loads and processes the CityscapesFoggy dataset for dehazing tasks.

    Args:
        root: Root directory path. Default is ``default_root_dir``.
        *args: Additional args for parent class.
        **kwargs: Additional kwargs for parent class.
    
    Raises:
        FileNotFoundError: If ``root``/cityscapes directory does not exist.
    """
    
    tasks : list[Task]  = [Task.DEHAZE]
    splits: list[Split] = [Split.TRAIN, Split.VAL, Split.TEST]
    datapoint_attrs     = DatapointAttributes({
        "image"    : Image,
        "ref_image": Image,
        "semantic" : SemanticMask,  # gtFine
    })
    has_test_annotations: bool = True
    
    def __init__(self, root: core.Path, *args, **kwargs):
        root = core.Path(root)
        root = root / "cityscapes" if root.name != "cityscapes" else root
        if not root.is_dir():
            raise FileNotFoundError(f"[root] directory not found: [{root}].")

        super().__init__(root=root, *args, **kwargs)
    
    def list_data(self):
        """Lists foggy images, reference images, and semantic maps."""
        patterns = [self.root / self.split_str / "leftImg8bit_foggy"]
        
        images: list[Image] = []
        with core.create_progress_bar(disable=self.disable_pbar) as pbar:
            for pattern in patterns:
                paths = sorted(pattern.rglob("*"))
                desc  = f"Listing {self.__class__.__name__} {self.split_str} images"
                for path in pbar.track(sequence=paths, description=desc):
                    if path.is_image_file():
                        images.append(Image(path=path, root=pattern))
        
        ref_images: list[Image] = []
        with core.create_progress_bar(disable=self.disable_pbar) as pbar:
            desc = f"Listing {self.__class__.__name__} {self.split_str} reference images"
            for img in pbar.track(sequence=images, description=desc):
                path = img.path.replace(f"{os.sep}leftImg8bit_foggy{os.sep}", f"{os.sep}leftImg8bit{os.sep}")
                stem = path.stem.split("leftImg8bit")[0]
                path = path.parent / f"{stem}leftImg8bit{path.suffix}"
                ref_images.append(Image(path=path.image_file()))
        
        # Semantic segmentation maps
        semantic: list[SemanticMask] = []
        with core.create_progress_bar(disable=self.disable_pbar) as pbar:
            desc = f"Listing {self.__class__.__name__} {self.split_str} semantic maps"
            for img in pbar.track(sequence=ref_images, description=desc):
                path = img.path.replace(f"{os.sep}leftImg8bit{os.sep}", f"{os.sep}gtFine{os.sep}")
                semantic.append(
                    SemanticMask(
                        path  = path.image_file(),
                        root  = img.root,
                        flags = cv2.IMREAD_GRAYSCALE
                    )
                )
        
        self.datapoints["image"]     = images
        self.datapoints["ref_image"] = ref_images
        self.datapoints["semantic"]  = semantic


# ----- DataModule -----
@DATAMODULES.register(name="cityscapes_foggy")
class CityscapesFoggyDataModule(core.DataModule):
    """Manages CityscapesFoggy dataset for training, validation, and testing."""

    tasks: list[Task] = [Task.DERAIN]

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
            self.train = CityscapesFoggy(split=Split.TRAIN, **self.dataset_kwargs)
            self.val   = CityscapesFoggy(split=Split.VAL,   **self.dataset_kwargs)
        if stage in [None, "test"]:
            self.test  = CityscapesFoggy(split=Split.TEST,  **self.dataset_kwargs)

        self.get_classes()
        if self.can_log:
            self.summarize()
