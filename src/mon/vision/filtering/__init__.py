#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Filtering.

This package provides functions and classes for performing various linear and
non-linear filtering operations on 2D images. It means that for each pixel
location in the source image (normally, rectangular), its neighborhood is
considered and used to compute the response.

Another common feature of the functions and classes described in this section
is that, unlike simple arithmetic functions, they need to extrapolate values of
some non-existing pixels.
"""

from __future__ import annotations

import mon.vision.filtering.box_filter
import mon.vision.filtering.guided_filter
import mon.vision.filtering.sobel_filter
from mon.vision.filtering.box_filter import *
from mon.vision.filtering.guided_filter import *
from mon.vision.filtering.sobel_filter import *
