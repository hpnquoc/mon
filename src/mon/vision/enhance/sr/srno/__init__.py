#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""SRNO model for super-resolution.

References:
    - Paper: "Super-Resolution Neural Operator," CVPR 2023.
    - Code: https://github.com/2y7c3/Super-Resolution-Neural-Operator
"""

__all__ = [
    "SRNO",
]

from .model import SRNO
from .src.models import make
from .src.utils import make_coord
