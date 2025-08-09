#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""LL-UNet++ model for low-light image enhancement.

References:
    - Paper: "LL-UNet++:UNet++ Based Nested Skip Connections Network for Low-Light
      Image Enhancement," TCI 2024.
    - Code: https://github.com/xiwang-online/LLUnetPlusPlus
"""

__all__ = [
    "LLUnetPP",
]

import os
import sys

import box

import mon.nn as nn
from mon.constants import MLType, MODELS, Task
from mon.core import pathlib

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from llunetpp.model import NestedUNet

current_file = pathlib.Path(__file__).absolute()
current_dir  = current_file.parents[0]


@MODELS.register(name="llunet++", arch="llunet++")
class LLUnetPP(NestedUNet, nn.ModelMixin):
    """LL-UNet++ model for low-light image enhancement.
    
    References:
        - Paper: "LL-UNet++:UNet++ Based Nested Skip Connections Network for Low-Light
          Image Enhancement," TCI 2024.
        - Code: https://github.com/xiwang-online/LLUnetPlusPlus
    """
    
    arch     : str          = "llunet++"
    name     : str          = "llunet++"
    tasks    : list[Task]   = [Task.LLE]
    mltypes  : list[MLType] = [MLType.SUPERVISED]
    model_dir: pathlib.Path = current_dir
    zoo      : dict         = box.Box()
