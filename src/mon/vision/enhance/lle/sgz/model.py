#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements SGZ model for low-light image enhancement.

References:
    - Paper: "Semantic-Guided Zero-Shot Learning for Low-Light Image/Video
      Enhancement," WACV 2022.
    - Code: https://github.com/ShenZheng2000/Semantic-Guided-Low-Light-Image-Enhancement
"""

__all__ = [
    "SGZ",
]

import box

from mon.constants import MODELS
from mon.core import MLType, ModelMixin, Path, Task
from .src.modeling.model import enhance_net_nopool

current_file = Path(__file__).absolute()
current_dir  = current_file.parents[0]


@MODELS.register(name="sgz", arch="sgz")
class SGZ(enhance_net_nopool, ModelMixin):
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
    model_dir: Path         = current_dir
    zoo      : dict         = box.Box()
