#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""ROI.

This module implements the Region of Interest (ROI) class, which represents a
region of an image that is being processed.
"""

from __future__ import annotations

__all__ = [
	"ROI",
]

from typing import Any

import cv2
import numpy as np


class ROI:
	"""Region of Interest (ROI).
	
	Args:
		id_: ROI's identifier.
		points: A sequence of points that defines the region boundary. Each item
			is a tuple of ``(x, y)`` coordinate.
		offset: The offset value when determining whether a vehicle is inside
			the ROI.
	"""
	
	def __init__(
		self,
		id_   : int,
		points: np.ndarray,
		offset: float = -50,
		*args, **kwargs
	):
		self.id_    = id_
		self.points = points
		self.offset = offset
		
		if len(self.points) < 2:
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
	
	def is_bbox_in_roi(self, bbox_xyxy: np.ndarray) -> bool:
		"""Check whether the bounding box is inside the ROI or not."""
		d = self.distance_between_bbox_center_and_roi(bbox_xyxy=bbox_xyxy)
		return d >= self.offset
	
	def distance_between_bbox_and_roi(self, bbox_xyxy: np.ndarray, compute_distance: bool = True) -> int:
		"""Compute the distance between the bounding box and the ROI.
		
		Args:
			bbox_xyxy: The bounding box coordinates in ``XYXY`` format.
			compute_distance: Should calculate the distance from the bounding
				box coordinates to the ROI? Default: ``False``.
				
		Returns:
			positive if the bounding box is inside the ROI,
			zero if the bounding box is on the edge of the ROI, and
			negative if the bounding box is outside the ROI.
		"""
		tl = cv2.pointPolygonTest(self.points, (bbox_xyxy[0], bbox_xyxy[1]), compute_distance)
		tr = cv2.pointPolygonTest(self.points, (bbox_xyxy[2], bbox_xyxy[1]), compute_distance)
		br = cv2.pointPolygonTest(self.points, (bbox_xyxy[2], bbox_xyxy[3]), compute_distance)
		bl = cv2.pointPolygonTest(self.points, (bbox_xyxy[0], bbox_xyxy[3]), compute_distance)
		if tl > 0 and tr > 0 and br > 0 and bl > 0:
			return min(tl, tr, br, bl)
		elif tl < 0 and tr < 0 and br < 0 and bl < 0:
			return min(tl, tr, br, bl)
		else:
			return 0
	
	def distance_between_bbox_center_and_roi(self, bbox_xyxy: np.ndarray, compute_distance: bool = True) -> int:
		"""Compute the distance between the bounding box center and the ROI.
		
		Args:
			bbox_xyxy: The bounding box coordinates in ``XYXY`` format.
			compute_distance: Should calculate the distance from the center of
				the bounding box to the ROI? Default: ``False``.
		
		Returns:
			positive if the bounding box is inside the ROI,
			zero if the bounding box is on the edge of the ROI, and
			negative if the bounding box is outside the ROI.
		"""
		cx = (bbox_xyxy[0] + bbox_xyxy[2]) / 2
		cy = (bbox_xyxy[1] + bbox_xyxy[3]) / 2
		return int(cv2.pointPolygonTest(self.points, (cx, cy), compute_distance))
	
	def draw(self, image: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
		"""Draw the ROI on the image."""
		pts = self.points.reshape((-1, 1, 2))
		cv2.polylines(img=image, pts=[pts], isClosed=True, color=color, thickness=2)
		return image
