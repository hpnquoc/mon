#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""MOI.

This module implements the Movement of Interest (MOI) class.
"""

from __future__ import annotations

__all__ = [
	"MOI",
]

import abc
from typing import Any, Literal

import cv2
import numpy as np


class MOI(abc.ABC):
	"""Movement of Interest (MOI).
	
	Attributes:
		type: The type of the MOI. One of: [None, ``"trajectory"``, ``"line_strip"``].
			Default: None.
	
	Args:
		id_: MOI's identifier.
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
		
		self.type: Literal[None, "trajectory", "line_strip"] = None

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


class TrajectoryMOI(MOI):
	"""Movement of Interest defined by a reference trajectory."""
	
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.type = "trajectory"


class LineStripMOI(MOI):
	"""Movement of Interest defined by a line strip."""
	
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.type = "line_strip"
