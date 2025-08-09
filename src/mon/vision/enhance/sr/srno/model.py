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

import os
import sys

import box

import mon.nn as nn
from mon.constants import MLType, MODELS, Task
from mon.core import pathlib

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from srno.models import sronet

current_file = pathlib.Path(__file__).absolute()
current_dir  = current_file.parents[0]


@MODELS.register(name="srno", arch="srno")
class SRNO(sronet.SRNO, nn.ModelMixin):
    """SRNO model for super-resolution.
    
    References:
        - Paper: "Super-Resolution Neural Operator," CVPR 2023.
        - Code: https://github.com/2y7c3/Super-Resolution-Neural-Operator
    """
    
    arch     : str          = "srno"
    name     : str          = "srno"
    tasks    : list[Task]   = [Task.SR]
    mltypes  : list[MLType] = [MLType.SUPERVISED]
    model_dir: pathlib.Path = current_dir
    zoo      : dict         = box.Box()
