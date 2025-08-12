#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""HVI-CIDNet model for low-light image enhancement.

References:
    - Paper: "HVI: A New color space for Low-light Image Enhancement," CVPR 2025.
    - Code: https://github.com/Fediory/HVI-CIDNet
"""

__all__ = [
    "HVI_CIDNet",
]

import box

import mon.nn as nn
from mon.constants import MLType, MODELS, Task
from mon.core import pathlib
from .src.net.CIDNet import CIDNet

current_file = pathlib.Path(__file__).absolute()
current_dir  = current_file.parents[0]


@MODELS.register(name="hvi_cidnet", arch="hvi_cidnet")
class HVI_CIDNet(CIDNet, nn.ModelMixin):
    """HVI-CIDNet model for low-light image enhancement.
    
    References:
        - Paper: "HVI: A New color space for Low-light Image Enhancement," CVPR 2025.
        - Code: https://github.com/Fediory/HVI-CIDNet
    """
    
    arch     : str          = "hvi_cidnet"
    name     : str          = "hvi_cidnet"
    tasks    : list[Task]   = [Task.LLE]
    mltypes  : list[MLType] = [MLType.SUPERVISED]
    model_dir: pathlib.Path = current_dir
    zoo      : dict         = box.Box()
