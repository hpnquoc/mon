#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""MOI.

This module implements the Movement of Interest (MOI) class.
"""

__all__ = [
	"MOI",
	"PolygonMOI",
	"TrajectoryMOI",
	"assign_moi_to_moving_objects",
]

import _template
from typing import Any, Literal

import cv2
import numpy as np

import mon
from tss.constants import AppleRGB


class MOI(_template.ABC):
	"""Movement of Interest (MOI).
	
	Attributes:
		type: The type of the MOI. One of: [None, ``"trajectory"``, ``"polygon"``].
			Default: None.
	
	Args:
		id_: MOI's identifier.
		points: A sequence of points that defines the region boundary. Each item
			is a tuple of ``(x, y)`` coordinate.
		color: The color of the MOI for visualization. Default: ``AppleRGB.DARK_GRAY3``.
	"""
	
	def __init__(
		self,
		id_   : int,
		points: np.ndarray,
		color : tuple[int, int, int] = AppleRGB.DARK_GRAY3,
		*args, **kwargs
	):
		self.id_    = id_
		self.points = points
		self.color  = color
		self.type: Literal[None, "trajectory", "polygon"] = None
		
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
		# Assert
		if len(self._points) < 2:
			raise ValueError("Insufficient number of points in the MOI.")
		
	@_template.abstractmethod
	def draw(self, image: np.ndarray, color: tuple[int, int, int] = None) -> np.ndarray:
		"""Draw the MOI on the image."""
		pass
	

class TrajectoryMOI(MOI):
	"""Movement of Interest defined by a reference trajectory.
	
	Args:
		distance_function: The distance function name. Default: ``"hausdorff"``.
		distance_threshold: The maximum distance for counting with track.
			Default: ``300.0``.
		angle_threshold: The maximum angle for counting with track.
			Default: ``45.0``.
	"""
	
	def __init__(
		self,
		distance_function : str   = "hausdorff",
		distance_threshold: float = 300.0,
		angle_threshold   : float = 45.0,
		*args, **kwargs
	):
		super().__init__(*args, **kwargs)
		self.distance_threshold = distance_threshold
		self.angle_threshold    = angle_threshold
		
		self.type = "trajectory"
		self.distance_function = mon.get_distance_function(name=distance_function)
	
	@staticmethod
	def assign_moi_to_moving_objects(mois: list[TrajectoryMOI], moving_objects: list):
		"""Assign the MOI to moving objects.
		
		Args:
			mois: A list of MOIs.
			moving_objects: A list of moving objects.
		"""
		for mo in moving_objects:
			# Calculate distances between object track and all mois' tracks
			distances = []
			angles    = []
			for moi in mois:
				distances.append(moi.distance_with_track(track=mo.trajectory))
				angles.append(moi.angle_with_track(track=mo.trajectory))
			
			min_moi_uuid = None
			min_distance = None
			for i, (d, a) in enumerate(zip(distances, angles)):
				if d is None or a is None:
					continue
				if (min_distance is not None) and (min_distance < d):
					continue
				min_distance = d
				min_moi_uuid = mois[i].id_
			
			mo.moi_id = min_moi_uuid
	
	def angle_with_track(self, track: np.ndarray) -> float | None:
		"""Calculate the angle between object's track with the MOI's reference
		trajectory.
		
		Args:
			track: The object's trajectory as an array of points.
			
		Returns:
			If calculated angle > self.angle_threshold, return ``None``.
		"""
		angle = mon.angle_between_arrays(self.points, track)
		return None if (angle > self.angle_threshold) else angle
	
	def distance_with_track(self, track: np.ndarray) -> float | None:
		"""Calculate the distance between object's track with the MOI's reference
		trajectory.
		
		Args:
			track: The object's trajectory as an array of points.
			
		Returns:
			If calculated distance > self.distance_threshold, return ``None``.
		"""
		distance = mon.hausdorff_distance(self.points, track)
		return None if (distance > self.distance_threshold) else distance
	
	def draw(self, image: np.ndarray, color: tuple[int, int, int] = None) -> np.ndarray:
		"""Draw the MOI on the image."""
		color = color or self.color
		pts   = self.points.reshape((-1, 1, 2))
		# Line
		cv2.polylines(img=image, pts=[pts], isClosed=False, color=color, thickness=1, lineType=cv2.LINE_AA)
		# Arrow head
		cv2.arrowedLine(img=image, pt1=tuple(self.points[-2]), pt2=tuple(self.points[-1]), color=color, thickness=1, line_type=cv2.LINE_AA, tipLength=0.2)
		# Each point
		for i in range(len(self.points) - 1):
			cv2.circle(img=image, center=tuple(self.points[i]), radius=3, color=color, thickness=-1, lineType=cv2.LINE_AA)
		# ID
		cv2.putText(img=image, text=f"{self.id_}", fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.75, org=tuple(self.points[-1]), color=color, thickness=2)
		return image


class PolygonMOI(MOI):
	"""Movement of Interest defined by a polygon.
	
	Args:
		offset: The offset value when determining whether a vehicle is inside
			the ROI. Default: -50.
	"""
	
	def __init__(
		self,
		offset: float = -50,
		*args, **kwargs
	):
		super().__init__(*args, **kwargs)
		self.offset = offset
		self.type   = "polygon"
	
	@staticmethod
	def assign_moi_to_moving_objects(mois: list[PolygonMOI], moving_objects: list):
		"""Assign the MOI to moving objects.
		
		Args:
			mois: A list of MOIs.
			moving_objects: A list of moving objects.
		"""
		for mo in moving_objects:
			for moi in mois:
				if mo.moi_id is None and moi.is_bbox_in_moi(bbox=mo.current_bbox):
					mo.moi_id = moi.id_
					break
	
	def is_bbox_in_moi(self, bbox: np.ndarray) -> bool:
		"""Check whether the bounding box is inside the MOI or not."""
		# d = self.distance_between_bbox_and_moi(bbox_xyxy)
		d = self.distance_between_bbox_center_and_moi(bbox)
		return d >= self.offset
	
	def distance_between_bbox_and_moi(self, bbox: np.ndarray) -> float:
		"""Compute the distance between the bounding box and the MOI.
		
		Args:
			bbox: The bounding box coordinates in ``XYXY`` format.
			
		Returns:
			positive if the bounding box is inside the ROI,
			zero if the bounding box is on the edge of the ROI, and
			negative if the bounding box is outside the ROI.
		"""
		return mon.distance_between_bbox_and_polygon(bbox, self.points)
	
	def distance_between_bbox_center_and_moi(self, bbox: np.ndarray) -> float:
		"""Compute the distance between the bounding box center and the MOI.
		
		Args:
			bbox: The bounding box coordinates in ``XYXY`` format.
			
		Returns:
			positive if the bounding box is inside the ROI,
			zero if the bounding box is on the edge of the ROI, and
			negative if the bounding box is outside the ROI.
		"""
		return mon.distance_between_bbox_center_and_polygon(bbox, self.points)
	
	def draw(self, image: np.ndarray, color: tuple[int, int, int] = None) -> np.ndarray:
		"""Draw the MOI on the image."""
		color = color or self.color
		pts   = self.points.reshape((-1, 1, 2))
		# Polygon
		cv2.polylines(img=image, pts=[pts], isClosed=True, color=color, thickness=1, lineType=cv2.LINE_AA)
		# ID
		cv2.putText(img=image, text=f"{self.id_}", fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.75, org=tuple(self.points[-1]), color=color, thickness=2)
		return image


def assign_moi_to_moving_objects(
	mois          : list[MOI],
	moving_objects: list,
	type          : Literal[None, "trajectory", "polygon"] = "trajectory"
):
	"""Assign the MOI to moving objects.
	
	Args:
		mois: A list of MOIs.
		moving_objects: A list of moving objects.
		type: The type of MOI to assign. Default: ``"trajectory"``.
	"""
	if len(moving_objects) <= 0:
		return
	trajectory_mois = [m for m in mois if m.type == "trajectory"]
	polygon_mois    = [m for m in mois if m.type == "polygon"]
	
	if type in [None, "trajectory"]:
		TrajectoryMOI.assign_moi_to_moving_objects(mois=trajectory_mois, moving_objects=moving_objects)
	elif type in [None, "polygon"]:
		PolygonMOI.assign_moi_to_moving_objects(mois=polygon_mois, moving_objects=moving_objects)
	else:
		raise ValueError(f"Invalid MOI type: {type}.")
