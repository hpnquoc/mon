#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""PSENet model for low-light image enhancement.

References:
    - Paper: "PSENet: Progressive Self-Enhancement Network for Unsupervised
      Extreme-Light Image Enhancement," WACV 2023.
    - Code: https://github.com/VinAIResearch/PSENet-Image-Enhancement
"""

__all__ = [
    "PSENet",
]

import box

import mon.nn as nn
from mon.constants import MLType, MODELS, Task
from mon.core import pathlib
from .src.model import UnetTMO

current_file = pathlib.Path(__file__).absolute()
current_dir  = current_file.parents[0]


@MODELS.register(name="psenet", arch="psenet")
class PSENet(UnetTMO, nn.ModelMixin):
    """PSENet model for low-light image enhancement.
    
    References:
        - Paper: "PSENet: Progressive Self-Enhancement Network for Unsupervised
          Extreme-Light Image Enhancement," WACV 2023.
        - Code: https://github.com/VinAIResearch/PSENet-Image-Enhancement
    """
    
    arch     : str          = "psenet"
    name     : str          = "psenet"
    tasks    : list[Task]   = [Task.LLE]
    mltypes  : list[MLType] = [MLType.UNSUPERVISED]
    model_dir: pathlib.Path = current_dir
    zoo      : dict         = box.Box()
