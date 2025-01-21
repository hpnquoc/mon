#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Detection.

This module implements the detection class, serving as a bridge between the
detector and the tracker. Additionally, a moving object is represented as a
list of sequential detections.
"""

from __future__ import annotations

__all__ = [
	"Detection",
]

import uuid
from timeit import default_timer as timer
from typing import Any

import cv2
import numpy as np

import mon
from tss.globals import AppleRGB, ID2CLASS


class Detection:
	"""The detection class is an interface that converts detected results from
	detectors to a standard and unified format for the tracker.
	
	Args:
		bbox: The bounding box in ``XYXY`` format.
		confidence: The confidence score of the detection. Default: ``0.0``.
		class_id: The class ID of the detection. Default: ``None`` or ``0`` means ``"unidentified"``.
		id_: The unique ID of the detection. Default: ``uuid.uuid4()``.
		frame_id: The frame ID of the detection. Default: ``None``.
		roi_id: The ROI ID of the detection. Default: ``None``.
		timestamp: The timestamp when the detection is created. Default: ``timer()``.
	"""
	
	def __init__(
		self,
		bbox      : np.ndarray,
		confidence: float           = 0.0,
		class_id  : int             = 0,
		id_       : int | uuid.UUID = uuid.uuid4(),
		frame_id  : int             = None,
		roi_id    : int             = None,
		timestamp : float           = timer(),
	):
		self.bbox		= bbox
		self.confidence = confidence
		self.class_id	= class_id
		self.id_		= id_
		self.frame_id	= frame_id
		self.roi_id		= roi_id
		self.timestamp	= timestamp
	
	@property
	def bbox(self) -> np.ndarray:
		return self._bbox
	
	@bbox.setter
	def bbox(self, bbox: Any):
		if isinstance(bbox, list):
			self._bbox = np.array(bbox, np.int32)
		elif isinstance(bbox, np.ndarray):
			self._bbox = bbox
		else:
			raise ValueError(f"`points` must be a `numpy.ndarray`, but got {type(bbox)}.")
		# Assert
		if len(self._bbox) != 4:
			raise ValueError(f"`bbox` must be in `XYXY` format, but got: {self._bbox}.")
			
	@property
	def bbox_center(self):
		"""Get the center of the bounding box."""
		return mon.bbox_center(bbox=self.bbox)
	
	@property
	def bbox_tl(self):
		"""Get the top-left corner of the bounding box."""
		return self.bbox[0:2]
	
	@property
	def class_id(self) -> int:
		"""Get the class ID of the detection."""
		return self._class_id
	
	@class_id.setter
	def class_id(self, class_id: int):
		if class_id is None:
			self._class_id = 0
		else:
			self._class_id = class_id
		# Assert
		if self._class_id not in ID2CLASS:
			raise ValueError(f"`class_id` must be one of: {ID2CLASS.keys()}, but got: {self._class_id}")
	
	@property
	def label(self) -> dict:
		"""Get the label of the detection."""
		if self.class_id not in ID2CLASS:
			raise ValueError(f"`class_id` must be one of: {ID2CLASS.keys()}, but got: {self.class_id}")
		return ID2CLASS.get(self.class_id)
		
	@property
	def color(self) -> tuple[int, int, int]:
		"""Get the color of the detection."""
		return self.label.get("color", AppleRGB.GRAY.value)
	
	def draw(
		self,
		image     : np.ndarray,
		draw_label: bool = True,
		color     : tuple[int, int, int] = None
	) -> np.ndarray:
		"""Draw the detection on the image."""
		color = color or self.color
		bbox  = self.bbox
		cv2.rectangle(img=image, pt1=(bbox[0], bbox[1]), pt2=(bbox[2], bbox[3]), color=color, thickness=2)
		# Label
		if draw_label:
			name = self.label.get("name")
			font = cv2.FONT_HERSHEY_SIMPLEX
			org  = (self.bbox_tl[0] + 5, self.bbox_tl[1])
			cv2.putText(img=image, text=name, fontFace=font, fontScale=1.0, org=org, color=color, thickness=2)
