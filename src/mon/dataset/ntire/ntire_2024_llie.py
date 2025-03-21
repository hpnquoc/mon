#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""NTIRE 2024 LLIE Challenge.

This module implements datasets and datamodules for NTIRE 2024 LLIE Challenge.

References:
	https://codalab.lisn.upsaclay.fr/competitions/17640
"""

from __future__ import annotations

__all__ = [
	"NTIRE2024LLIE",
	"NTIRE2024LLIEDataModule",
]

from typing import Literal

from mon import core
from mon.dataset import dtype
from mon.globals import DATA_DIR, DATAMODULES, DATASETS, Split, Task

console             = core.console
default_root_dir    = DATA_DIR / "ntire"
DataModule          = dtype.DataModule
DatapointAttributes = dtype.DatapointAttributes
DepthMapAnnotation  = dtype.DepthMapAnnotation
ImageAnnotation     = dtype.ImageAnnotation
MultimodalDataset   = dtype.MultimodalDataset


# region Dataset

@DATASETS.register(name="ntire_2024_llie")
class NTIRE2024LLIE(MultimodalDataset):
	"""
	References:
		https://codalab.lisn.upsaclay.fr/competitions/17640
	"""
	
	tasks : list[Task]  = [Task.LLIE]
	splits: list[Split] = [Split.TRAIN, Split.VAL, Split.TEST]
	datapoint_attrs     = DatapointAttributes({
		"image"    : ImageAnnotation,
		"ref_image": ImageAnnotation,
	})
	has_test_annotations: bool = False
	
	def __init__(self, root: core.Path = default_root_dir, *args, **kwargs):
		root = root / "ntire_2024_llie" if root.name != "ntire_2024_llie" else root
		if not root.is_dir():
			raise FileNotFoundError(f"Directory not found: {root}.")
		# Initialize
		super().__init__(root=root, *args, **kwargs)
	
	def get_data(self):
		if self.split in [Split.TRAIN]:
			patterns = [
				self.root / "train" / "image",
			]
		elif self.split in [Split.VAL]:
			patterns = [
				self.root / "val" / "image",
			]
		elif self.split in [Split.TEST]:
			patterns = [
				self.root / "test" / "image",
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

@DATAMODULES.register(name="ntire_2024_llie")
class NTIRE2024LLIEDataModule(DataModule):
	"""
	References:
		https://codalab.lisn.upsaclay.fr/competitions/17640
	"""
	
	tasks: list[Task] = [Task.LLIE]
	
	def prepare_data(self, *args, **kwargs):
		pass
	
	def setup(self, stage: Literal["train", "test", "predict", None] = None):
		if self.can_log:
			console.log(f"Setup [red]{self.__class__.__name__}[/red].")
		
		if stage in [None, "train"]:
			self.train = NTIRE2024LLIE(split=Split.TRAIN, **self.dataset_kwargs)
			self.val   = NTIRE2024LLIE(split=Split.TRAIN, **self.dataset_kwargs)
		if stage in [None, "test"]:
			self.test  = NTIRE2024LLIE(split=Split.VAL,   **self.dataset_kwargs)
		
		self.get_classlabels()
		if self.can_log:
			self.summarize()
	
# endregion
