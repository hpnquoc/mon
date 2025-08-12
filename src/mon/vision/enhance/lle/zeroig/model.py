#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""ZERO-IG model for low-light image enhancement.

References:
    - Paper: "Zero-Shot Illumination-Guided Joint Denoising and Adaptive
      Enhancement for Low-Light Images," CVPR 2024.
    - Code: https://github.com/Doyle59217/ZeroIG
"""

__all__ = [
    "ZERO_IG",
    "ZERO_IG_Finetune",
]

import box

import mon.nn as nn
from mon.constants import MLType, MODELS, Task
from mon.core import pathlib
from .src.model import Finetunemodel, Network

current_file = pathlib.Path(__file__).absolute()
current_dir  = current_file.parents[0]


@MODELS.register(name="zeroig", arch="zeroig")
class ZERO_IG(Network, nn.ModelMixin):
    """ZERO-IG model for low-light image enhancement.
    
    References:
        - Paper: "Zero-Shot Illumination-Guided Joint Denoising and Adaptive
          Enhancement for Low-Light Images," CVPR 2024.
        - Code: https://github.com/Doyle59217/ZeroIG
    """
    
    arch     : str          = "zeroig"
    name     : str          = "zeroig"
    tasks    : list[Task]   = [Task.LLE]
    mltypes  : list[MLType] = [MLType.UNSUPERVISED]
    model_dir: pathlib.Path = current_dir
    zoo      : dict         = box.Box()


class ZERO_IG_Finetune(Finetunemodel):
    """ZERO-IG model for low-light image enhancement.
    
    References:
        - Paper: "Zero-Shot Illumination-Guided Joint Denoising and Adaptive
          Enhancement for Low-Light Images," CVPR 2024.
        - Code: https://github.com/Doyle59217/ZeroIG
    """
    pass
