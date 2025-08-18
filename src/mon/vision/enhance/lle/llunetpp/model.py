#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements LL-UNet++ model for low-light image enhancement.

References:
    - Paper: "LL-UNet++:UNet++ Based Nested Skip Connections Network for Low-Light
      Image Enhancement," TCI 2024.
    - Code: https://github.com/xiwang-online/LLUnetPlusPlus
"""

__all__ = [
    "LLUnetPP",
]

import box

from mon.constants import MODELS
from mon.core import MLType, ModelMixin, Path, Task
from .src.model import NestedUNet

current_file = Path(__file__).absolute()
current_dir  = current_file.parents[0]


@MODELS.register(name="llunet++", arch="llunet++")
class LLUnetPP(NestedUNet, ModelMixin):
    """LL-UNet++ model for low-light image enhancement.
    
    References:
        - Paper: "LL-UNet++:UNet++ Based Nested Skip Connections Network for Low-Light
          Image Enhancement," TCI 2024.
        - Code: https://github.com/xiwang-online/LLUnetPlusPlus
    """
    
    arch     : str          = "llunet++"
    name     : str          = "llunet++"
    tasks    : list[Task]   = [Task.LLE]
    mltypes  : list[MLType] = [MLType.SUPERVISED]
    model_dir: Path         = current_dir
    zoo      : dict         = box.Box()
