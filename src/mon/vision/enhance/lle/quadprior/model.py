#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""QuadPrior model for low-light image enhancement.

References:
    - Paper: "Zero-Reference Low-Light Enhancement via Physical Quadruple
      Priors," CVPR 2024.
    - Code: https://github.com/daooshee/QuadPrior
"""

__all__ = [
    "QuadPrior",
]

import os
import sys

import box

import mon.nn as nn
from mon.constants import MLType, MODELS, Task
from mon.core import pathlib

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from quadprior.cldm.cldm import ControlLDM

current_file = pathlib.Path(__file__).absolute()
current_dir  = current_file.parents[0]


@MODELS.register(name="quadprior", arch="quadprior")
class QuadPrior(ControlLDM, nn.ModelMixin):
    """QuadPrior model for low-light image enhancement.
    
    References:
        - Paper: "Zero-Reference Low-Light Enhancement via Physical Quadruple
          Priors," CVPR 2024.
        - Code: https://github.com/daooshee/QuadPrior
    """
    
    arch     : str          = "quadprior"
    name     : str          = "quadprior"
    tasks    : list[Task]   = [Task.LLE]
    mltypes  : list[MLType] = [MLType.UNSUPERVISED]
    model_dir: pathlib.Path = current_dir
    zoo      : dict         = box.Box()
