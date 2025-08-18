#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements loss functions for neural network training."""

# noinspection PyUnusedImports
from torch.nn.modules.loss import *  # Expose all loss functions from ``torch``.

from .base import *
from .image import *
from .basic import *
