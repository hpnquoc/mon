#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements annotation types for vision tasks.

An annotation refers to metadata or labels associated with vision data that provide
context, meaning, or ground truth for training, evaluation, or interpretation.
Annotations describe specific aspects of the visual data, such as object locations,
categories, or semantic regions, and are typically created manually or semi-automatically.
"""

from mon.vision.types.annotations.bbox import *
from mon.vision.types.annotations.contour import *
from mon.vision.types.annotations.depth import *
from mon.vision.types.annotations.image import *
from mon.vision.types.annotations.label_map import *
