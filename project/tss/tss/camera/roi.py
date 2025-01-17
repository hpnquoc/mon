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

from typing import Sequence


class ROI:
	"""Region of Interest (ROI).
	
	Args:
		id_: ROI's identifier.
		region: Region coordinates as a list of tuples, where each tuple is a
			pair of (x, y) coordinates.
	"""
	
	def __init__(
		self,
		id_   : int,
		region: list[tuple[int, int]] | Sequence,
	):
		self.id_    = id_
		self.region = region
