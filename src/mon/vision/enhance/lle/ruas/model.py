#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""RUAS model for low-light image enhancement.

References:
    - Paper: "Retinex-inspired Unrolling with Cooperative Prior Architecture
      Search for Low-light Image Enhancement," 2021.
    - Code: https://github.com/KarelZhang/RUAS
"""

__all__ = [
    "RUAS",
]

import os
import sys

import box

import mon.nn as nn
from mon.constants import MLType, MODELS, Task
from mon.core import pathlib

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from ruas.model import Network

current_file = pathlib.Path(__file__).absolute()
current_dir  = current_file.parents[0]


@MODELS.register(name="ruas", arch="ruas")
class RUAS(Network, nn.ModelMixin):
    """RUAS model for low-light image enhancement.
    
    References:
        - Paper: "Retinex-inspired Unrolling with Cooperative Prior Architecture
          Search for Low-light Image Enhancement," 2021.
        - Code: https://github.com/KarelZhang/RUAS
    """
    
    arch     : str          = "ruas"
    name     : str          = "ruas"
    tasks    : list[Task]   = [Task.LLE]
    mltypes  : list[MLType] = [MLType.UNSUPERVISED]
    model_dir: pathlib.Path = current_dir
    zoo      : dict         = box.Box()
