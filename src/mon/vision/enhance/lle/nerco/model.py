#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""NeRCo model for low-light image enhancement.

References:
    - Paper: "Implicit Neural Representation for Cooperative Low-light
      Image Enhancement," ICCV 2023.
    - Code: https://github.com/Ysz2022/NeRCo
"""

__all__ = [
    "NeRCo",
]

import os
import sys

import box

import mon.nn as nn
from mon.constants import MLType, MODELS, Task
from mon.core import pathlib

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from nerco.models.nerco_model import NeRComodel

current_file = pathlib.Path(__file__).absolute()
current_dir  = current_file.parents[0]


@MODELS.register(name="nerco", arch="nerco")
class NeRCo(NeRComodel, nn.ModelMixin):
    """NeRCo model for low-light image enhancement.
    
    References:
        - Paper: "Implicit Neural Representation for Cooperative Low-light
          Image Enhancement," ICCV 2023.
        - Code: https://github.com/Ysz2022/NeRCo
    """
    
    arch     : str          = "nerco"
    name     : str          = "nerco"
    tasks    : list[Task]   = [Task.LLE]
    mltypes  : list[MLType] = [MLType.UNSUPERVISED]
    model_dir: pathlib.Path = current_dir
    zoo      : dict         = box.Box()
