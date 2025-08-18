#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements PSENet model for low-light image enhancement.

References:
    - Paper: "PSENet: Progressive Self-Enhancement Network for Unsupervised
      Extreme-Light Image Enhancement," WACV 2023.
    - Code: https://github.com/VinAIResearch/PSENet-Image-Enhancement
"""

__all__ = [
    "PSENet",
]

import box

from mon.constants import MODELS
from mon.core import MLType, ModelMixin, Path, Task
from .src.model import UnetTMO

current_file = Path(__file__).absolute()
current_dir  = current_file.parents[0]


@MODELS.register(name="psenet", arch="psenet")
class PSENet(UnetTMO, ModelMixin):
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
    model_dir: Path         = current_dir
    zoo      : dict         = box.Box()
