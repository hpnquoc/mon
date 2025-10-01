#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Supports ML/DL research, built on top of ``PyTorch``.

Notes:
    - In this package, we follow the same coding conventions as PyTorch to
      maintain consistency.
    - If you don't know what to do, just look at the PyTorch source code.
"""

__all__ = [
    "BaseLoss",
    "ModelMixin",
]

# noinspection PyUnusedImports
from torch.nn import *  # Export all modules from ``torch.nn``

from mon.core.nn.modules import (
    activation as activation,
    attention as attention,
    cnn as cnn,
    container as container,
    dropout as dropout,
    fusion as fusion,
    inr as inr,
    linear as linear,
    misc as misc,
    norm as norm,
    padding as padding,
    pooling as pooling,
    rnn as rnn,
    transformer as transformer,
)
from .losses import *
from .metrics import *
from .model import *
from .modules import *
from .optims import *
