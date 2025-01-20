#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""ROI.

This module implements the Region of Interest (ROI) class, which represents a
region of an image that is being processed.
"""

from __future__ import annotations

__all__ = [
	"ROI",
	"assign_roi_to_detections",
]

from typing import Any

import cv2
import numpy as np

import mon


class ROI:
	"""Region of Interest (ROI).
	
	Args:
		id_: ROI's identifier.
		points: A sequence of points that defines the region boundary. Each item
			is a tuple of ``(x, y)`` coordinate.
		offset: The offset value when determining whether a vehicle is inside
			the ROI. Default: -50.
		color: The color of the ROI for visualization.
			Default: ``(52, 199, 89)`` - Apple's Green.
	"""
	
	def __init__(
		self,
		id_   : int,
		points: np.ndarray,
		offset: float = -50,
		color : tuple[int, int, int] = (52, 199, 89),  # Apple's Green
		*args, **kwargs
	):
		self.id_    = id_
		self.points = points
		self.offset = offset
		self.color  = color
		
		if self.points is None or len(self.points) < 2:
			raise ValueError("Insufficient number of points in the ROI.")
	
	@property
	def points(self) -> np.ndarray:
		return self._points
	
	@points.setter
	def points(self, points: Any):
		if isinstance(points, list):
			self._points = np.array(points, np.int32)
		elif isinstance(points, np.ndarray):
			self._points = points
		else:
			raise ValueError(f"`points` must be a `numpy.ndarray`, but got {type(points)}.")
	
	def is_bbox_in_roi(self, bbox: np.ndarray) -> bool:
		"""Check whether the bounding box is inside the ROI or not."""
		# d = self.distance_between_bbox_and_roi(bbox_xyxy)
		d = self.distance_between_bbox_center_and_roi(bbox)
		return d >= self.offset
	
	def distance_between_bbox_and_roi(self, bbox: np.ndarray) -> float:
		"""Compute the distance between the bounding box and the ROI.
		
		Args:
			bbox: The bounding box coordinates in ``XYXY`` format.
			
		Returns:
			positive if the bounding box is inside the ROI,
			zero if the bounding box is on the edge of the ROI, and
			negative if the bounding box is outside the ROI.
		"""
		return mon.distance_between_bbox_and_polygon(bbox, self.points)
		
	def distance_between_bbox_center_and_roi(self, bbox: np.ndarray) -> float:
		"""Compute the distance between the bounding box center and the ROI.
		
		Args:
			bbox: The bounding box coordinates in ``XYXY`` format.
			
		Returns:
			positive if the bounding box is inside the ROI,
			zero if the bounding box is on the edge of the ROI, and
			negative if the bounding box is outside the ROI.
		"""
		return mon.distance_between_bbox_center_and_polygon(bbox, self.points)
	
	def draw(self, image: np.ndarray, color: tuple[int, int, int] = None) -> np.ndarray:
		"""Draw the ROI on the image."""
		color = color or self.color
		pts   = self.points.reshape((-1, 1, 2))
		cv2.polylines(img=image, pts=[pts], isClosed=True, color=color, thickness=2)
		return image


def assign_roi_to_detections(rois: list[ROI], detections: list):
	"""Assign the ROI to detections.
	
	Args:
		rois: A list of ROIs.
		detections: A list of :obj:`Detection` objects.
	"""
	for d in detections:
		for roi in rois:
			if roi.is_bbox_in_roi(bbox=d.bbox):
				d.roi_id = roi.id_
				break
