#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements HVI-CIDNet model for low-light image enhancement.

References:
    - Paper: "HVI: A New color space for Low-light Image Enhancement," CVPR 2025.
    - Code: https://github.com/Fediory/HVI-CIDNet
"""

__all__ = [
    "HVI_CIDNet",
]

import box

from mon.constants import MODELS
from mon.core import MLType, ModelMixin, Path, Task
from .src.net.CIDNet import CIDNet

current_file = Path(__file__).absolute()
current_dir  = current_file.parents[0]


@MODELS.register(name="hvi_cidnet", arch="hvi_cidnet")
class HVI_CIDNet(CIDNet, ModelMixin):
    """HVI-CIDNet model for low-light image enhancement.
    
    References:
        - Paper: "HVI: A New color space for Low-light Image Enhancement," CVPR 2025.
        - Code: https://github.com/Fediory/HVI-CIDNet
    """
    
    arch     : str          = "hvi_cidnet"
    name     : str          = "hvi_cidnet"
    tasks    : list[Task]   = [Task.LLE]
    mltypes  : list[MLType] = [MLType.SUPERVISED]
    model_dir: Path         = current_dir
    zoo      : dict         = box.Box()
