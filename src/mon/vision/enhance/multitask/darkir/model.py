#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""DarkIR model for low-light deblurring.

References:
    - Paper: "DarkIR: Robust Low-Light Image Restoration," CVPR 2025.
    - Code: https://github.com/cidautai/DarkIR
"""

__all__ = [
    "DarkIR",
]

import os
import sys

import box

import mon.nn as nn
from mon.constants import MLType, MODELS, Task
from mon.core import pathlib

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from darkir import archs

current_file = pathlib.Path(__file__).absolute()
current_dir  = current_file.parents[0]


class DarkIR(archs.DarkIR, nn.ModelMixin):
    """DarkIR model for low-light deblurring.
    
    References:
        - Paper: "DarkIR: Robust Low-Light Image Restoration," CVPR 2025.
        - Code: https://github.com/cidautai/DarkIR
    """
    
    arch     : str          = "darkir"
    name     : str          = "darkir"
    tasks    : list[Task]   = [Task.LLE, Task.DEBLUR]
    mltypes  : list[MLType] = [MLType.SUPERVISED]
    model_dir: pathlib.Path = current_dir
    zoo      : dict         = box.Box()


MODELS.register(name="darkir_m", arch="darkir", module=DarkIR)
MODELS.register(name="darkir_l", arch="darkir", module=DarkIR)
