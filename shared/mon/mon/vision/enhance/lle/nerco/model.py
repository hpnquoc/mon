#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements NeRCo model for low-light image enhancement.

References:
    - Paper: "Implicit Neural Representation for Cooperative Low-light
      Image Enhancement," ICCV 2023.
    - Code: https://github.com/Ysz2022/NeRCo
"""

__all__ = [
    "NeRCo",
]

import box

from mon.constants import MODELS
from mon.core import MLType, ModelMixin, Path, Task
from .src.models.nerco_model import NeRComodel

current_file = Path(__file__).absolute()
current_dir  = current_file.parents[0]


@MODELS.register(name="nerco", arch="nerco")
class NeRCo(NeRComodel, ModelMixin):
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
    model_dir: Path         = current_dir
    zoo      : dict         = box.Box()
