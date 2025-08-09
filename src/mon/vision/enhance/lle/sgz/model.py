#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""SGZ model for low-light image enhancement.

References:
    - Paper: "Semantic-Guided Zero-Shot Learning for Low-Light Image/Video
      Enhancement," WACV 2022.
    - Code: https://github.com/ShenZheng2000/Semantic-Guided-Low-Light-Image-Enhancement
"""

__all__ = [
    "SGZ",
]

import os
import sys

import box

import mon.nn as nn
from mon.constants import MLType, MODELS, Task
from mon.core import pathlib

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from sgz.modeling.model import enhance_net_nopool

current_file = pathlib.Path(__file__).absolute()
current_dir  = current_file.parents[0]


@MODELS.register(name="sgz", arch="sgz")
class SGZ(enhance_net_nopool, nn.ModelMixin):
    """SGZ model for low-light image enhancement.
    
    References:
        - Paper: "Semantic-Guided Zero-Shot Learning for Low-Light Image/Video
          Enhancement," WACV 2022.
        - Code: https://github.com/ShenZheng2000/Semantic-Guided-Low-Light-Image-Enhancement
    """
    
    arch     : str          = "sgz"
    name     : str          = "sgz"
    tasks    : list[Task]   = [Task.LLE]
    mltypes  : list[MLType] = [MLType.UNSUPERVISED]
    model_dir: pathlib.Path = current_dir
    zoo      : dict         = box.Box()
