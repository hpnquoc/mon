#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""DepthAnything.

This module implements a wrapper for DepthAnything models. It provides a
simple interface for loading pre-trained models and performing inference.

Notice:
This is the first example of using a third-party package in the `mon` package.
Why? Because reimplementing all of :obj:`depth_anything_v2` is a lot of work and
is not a smart idea.
"""

from __future__ import annotations

import mon.vision.depth.depth_anything.depth_anything
from mon.vision.depth.depth_anything.depth_anything import *
