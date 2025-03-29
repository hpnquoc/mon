#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements annotation types for vision tasks.

An annotation refers to metadata or labels associated with vision data that provide
context, meaning, or ground truth for training, evaluation, or interpretation.
Annotations describe specific aspects of the visual data, such as object locations,
categories, or semantic regions, and are typically created manually or semi-automatically.
"""

from __future__ import annotations

import mon.vision.dtype.annotation.bbox
import mon.vision.dtype.annotation.contour
import mon.vision.dtype.annotation.label_map
from mon.vision.dtype.annotation.bbox import *
from mon.vision.dtype.annotation.contour import *
from mon.vision.dtype.annotation.label_map import *
