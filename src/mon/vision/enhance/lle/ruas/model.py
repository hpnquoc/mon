#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements RUAS model for low-light image enhancement.

References:
    - Paper: "Retinex-inspired Unrolling with Cooperative Prior Architecture
      Search for Low-light Image Enhancement," 2021.
    - Code: https://github.com/KarelZhang/RUAS
"""

__all__ = [
    "RUAS",
]

import box

from mon.constants import MODELS
from mon.core import MLType, ModelMixin, Path, Task
from .src.model import Network

current_file = Path(__file__).absolute()
current_dir  = current_file.parents[0]


@MODELS.register(name="ruas", arch="ruas")
class RUAS(Network, ModelMixin):
    """RUAS model for low-light image enhancement.
    
    References:
        - Paper: "Retinex-inspired Unrolling with Cooperative Prior Architecture
          Search for Low-light Image Enhancement," 2021.
        - Code: https://github.com/KarelZhang/RUAS
    """
    
    arch     : str          = "ruas"
    name     : str          = "ruas"
    tasks    : list[Task]   = [Task.LLE]
    mltypes  : list[MLType] = [MLType.UNSUPERVISED]
    model_dir: Path         = current_dir
    zoo      : dict         = box.Box()
