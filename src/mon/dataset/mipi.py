#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""MIPI Challenges.

This module implements datasets and datamodules for MIPI challenges.

References:
	https://mipi-challenge.org/MIPI2024/index.html
"""

from __future__ import annotations

__all__ = [
	"MIPI2024Flare",
	"MIPI2024FlareDataModule",
]

from typing import Literal

from mon import core
from mon.dataset import dtype
from mon.globals import DATA_DIR, DATAMODULES, DATASETS, Split, Task

console             = core.console
default_root_dir    = DATA_DIR / "mipi"
DataModule          = dtype.DataModule
DatapointAttributes = dtype.DatapointAttributes
DepthMapAnnotation  = dtype.DepthMapAnnotation
ImageAnnotation     = dtype.ImageAnnotation
MultimodalDataset   = dtype.MultimodalDataset


# region Dataset

@DATASETS.register(name="mipi2024_flare")
class MIPI2024Flare(MultimodalDataset):
	"""Nighttime Flare Removal dataset used in MIPI 2024 Challenge.
	
	References:
		https://mipi-challenge.org/MIPI2024/index.html
	"""
	
	tasks : list[Task]  = [Task.NIGHTTIME]
	splits: list[Split] = [Split.TRAIN, Split.VAL, Split.TEST]
	datapoint_attrs     = DatapointAttributes({
		"image"    : ImageAnnotation,
		"ref_image": ImageAnnotation,
	})
	has_test_annotations: bool = False
	
	def __init__(self, root: core.Path = default_root_dir, *args, **kwargs):
		super().__init__(root=root, *args, **kwargs)
	
	def get_data(self):
		if self.split in [Split.TRAIN]:
			patterns = [
				self.root / "mipi24_flare" / "train" / "image",
			]
		elif self.split in [Split.VAL]:
			patterns = [
				self.root / "mipi24_flare" / "val" / "image",
			]
		elif self.split in [Split.TEST]:
			patterns = [
				self.root / "mipi24_flare" / "test" / "image",
			]
		else:
			raise ValueError
		
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
		
		
# region DataModule

@DATAMODULES.register(name="mipi2024_flare")
class MIPI2024FlareDataModule(DataModule):
	"""Nighttime Flare Removal datamodule used in MIPI 2024 Challenge.
	
	References:
		https://mipi-challenge.org/MIPI2024/index.html
	"""
	
	tasks: list[Task] = [Task.NIGHTTIME]
	
	def prepare_data(self, *args, **kwargs):
		pass
	
	def setup(self, stage: Literal["train", "test", "predict", None] = None):
		if self.can_log:
			console.log(f"Setup [red]{self.__class__.__name__}[/red].")
		
		if stage in [None, "train"]:
			self.train = MIPI2024Flare(split=Split.TRAIN, **self.dataset_kwargs)
			self.val   = MIPI2024Flare(split=Split.VAL, **self.dataset_kwargs)
		if stage in [None, "test"]:
			self.test  = MIPI2024Flare(split=Split.VAL, **self.dataset_kwargs)
		
		self.get_classlabels()
		if self.can_log:
			self.summarize()
	
# endregion
