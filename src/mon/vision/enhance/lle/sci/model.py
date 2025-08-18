#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements SCI model for low-light image enhancement.

References:
    - Paper: "Toward Fast, Flexible, and Robust Low-Light Image Enhancement,"
      CVPR 2022.
    - Code: https://github.com/vis-opt-group/SCI
"""

__all__ = [
    "SCI",
]

import box

from mon.constants import MODELS
from mon.core import MLType, ModelMixin, Path, Task
from .src.model import Finetunemodel

current_file = Path(__file__).absolute()
current_dir  = current_file.parents[0]


@MODELS.register(name="sci", arch="sci")
class SCI(Finetunemodel, ModelMixin):
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
    model_dir: Path         = current_dir
    zoo      : dict         = box.Box()
