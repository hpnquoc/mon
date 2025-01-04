#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Datasets Templates.

This module implements base classes for all datasets.

For transformation operations, we use
`albumentations <https://albumentations.ai/docs/api_reference/full_reference>`__
"""

from __future__ import annotations

import mon.dataset.dtype.dataset.base
import mon.dataset.dtype.dataset.image
import mon.dataset.dtype.dataset.video
from mon.dataset.dtype.dataset.base import *
from mon.dataset.dtype.dataset.image import *
from mon.dataset.dtype.dataset.video import *
