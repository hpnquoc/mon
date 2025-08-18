#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Object.

This module implements object classes.
"""

__all__ = [
	"GeneralObject",
	"MovingObject"
]

import _template
import uuid
from collections import Counter

import cv2
import numpy as np

import mon
from tss.dtype import detection as D
from tss.constants import AppleRGB, ID2CLASS

Detection = D.Detection


class GeneralObject(_template.ABC):
	"""A base class for all objects."""
	pass


class MovingObject(GeneralObject):
	"""Moving object class describes an object that moves in the scene.
	
	Args:
		id_: The unique ID of the object.
		detections: A sequential list of :obj:`Detection` of the object.
	"""
	
	def __init__(
		self,
		id_       : int | uuid.UUID = uuid.uuid4(),
		detections: list[Detection] = [],
	):
		super().__init__()
		self.id_       = id_
		self.instances = detections
	
	@property
	def trajectory(self) -> np.ndarray | None:
		"""Get the trajectory of the object as a :obj:`numpy.ndarray` of center points."""
		return np.array([i.bbox_center for i in self.instances]) if len(self.instances) > 0 else None
	
	@property
	def travelled_distance(self) -> float:
		"""Get the travelled distance of the object."""
		t = self.trajectory
		return mon.distance_between_points(t[0], t[-1]) if t is not None else 0.0
	
	@property
	def first(self) -> Detection | None:
		return self.instances[0] if len(self.instances) > 0 else None
	
	@property
	def current(self) -> Detection | None:
		return self.instances[-1] if len(self.instances) > 0 else None
	
	@property
	def current_label(self) -> dict | None:
		return self.current.label if self.current is not None else None
		
	@property
	def majority_class_id(self) -> int | None:
		"""Get the majority class ID of the object."""
		if len(self.instances) <= 0:
			return None
		elif len(self.instances) == 1:
			return self.instances[0].class_id
		else:
			class_ids   = [instance.class_id for instance in self.instances]
			counter     = Counter(class_ids)
			most_common = counter.most_common(1)[0]  # Get the most common number and its frequency
			return most_common[0]
	
	@property
	def majority_label(self) -> dict | None:
		cls_id = self.majority_class_id
		return ID2CLASS.get(cls_id) if cls_id is not None else None
		
	@property
	def color(self) -> tuple[int, int, int]:
		if isinstance(self.majority_label, dict):
			return self.majority_label.get("color")
		else:
			return AppleRGB.GRAY.value
	
	def update(self, detection: Detection):
		"""Update the object with a new detection."""
		self.instances.append(detection)
	
	def draw(
		self,
		image          : np.ndarray,
		draw_label     : bool = True,
		draw_trajectory: bool = True,
		color          : tuple[int, int, int] = None,
	) -> np.ndarray:
		"""Draw the object on the image."""
		if len(self.instances) <= 0:
			return image
		color  = color or self.color
		# Bounding box
		bbox   = self.current.bbox
		center = self.current.bbox_center
		cv2.rectangle(img=image, pt1=(bbox[0], bbox[1]), pt2=(bbox[2], bbox[3]), color=color, thickness=2)
		cv2.circle(img=image, center=tuple(center), radius=3, thickness=-1, color=color)
		# Label
		if draw_label:
			bbox_tl = self.current.bbox_tl
			label   = self.majority_label
			org     = (bbox_tl[0] + 5, bbox_tl[1])
			cv2.putText(img=image, text=label["name"], fontFace= cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, org=org, color=color, thickness=2)
		# Trajectory
		if draw_trajectory:
			pts = self.trajectory.reshape((-1, 1, 2))
			cv2.polylines(img=image, pts=[pts], isClosed=False, color=color, thickness=2)
