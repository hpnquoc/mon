#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""SCI model for low-light image enhancement.

References:
    - Paper: "Toward Fast, Flexible, and Robust Low-Light Image Enhancement,"
      CVPR 2022.
    - Code: https://github.com/vis-opt-group/SCI
"""

__all__ = [
    "SCI",
]

import os
import sys

import box

import mon.nn as nn
from mon.constants import MLType, MODELS, Task
from mon.core import pathlib

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from sci.model import Finetunemodel

current_file = pathlib.Path(__file__).absolute()
current_dir  = current_file.parents[0]


@MODELS.register(name="sci", arch="sci")
class SCI(Finetunemodel, nn.ModelMixin):
    """SCI model for low-light image enhancement.
    
    References:
        - Paper: "Toward Fast, Flexible, and Robust Low-Light Image Enhancement,"
          CVPR 2022.
        - Code: https://github.com/vis-opt-group/SCI
    """
    
    arch     : str          = "sci"
    name     : str          = "sci"
    tasks    : list[Task]   = [Task.LLE]
    mltypes  : list[MLType] = [MLType.UNSUPERVISED]
    model_dir: pathlib.Path = current_dir
    zoo      : dict         = box.Box()
