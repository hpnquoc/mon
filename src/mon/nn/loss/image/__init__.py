#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements loss functions for images.

The categories align with common loss function roles in computer vision:
    - objective.py : image quality metrics (objective fidelity).
    - perceptual.py: perceptual losses (human-like perception).
    - structural.py: edge/structural regularization (detail preservation).
    - spatial.py   : spatial consistency (coherence across regions).
    - color.py     : color/illumination consistency (photometric accuracy).
"""

from .color import *
from .objective import *
from .perceptual import *
from .spatial import *
from .structural import *
