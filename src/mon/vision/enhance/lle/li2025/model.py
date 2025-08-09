#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Li2025 model for low-light image enhancement.

References:
    - Paper: "Interpretable Unsupervised Joint Denoising and Enhancement for
      Real-World low-light Scenarios," ICLR 2025.
    - Code: https://github.com/huaqlili/unsupervised-light-enhance-ICLR2025
"""

__all__ = [
    "Li2025",
]

import os
import sys

import box

import mon.nn as nn
from mon.constants import MLType, MODELS, Task
from mon.core import pathlib

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from src.net.lformer import net

current_file = pathlib.Path(__file__).absolute()
current_dir  = current_file.parents[0]


@MODELS.register(name="li2025", arch="li2025")
class Li2025(net, nn.ModelMixin):
    """Li2025 model for low-light image enhancement.
    
    References:
        - Paper: "Interpretable Unsupervised Joint Denoising and Enhancement for
          Real-World low-light Scenarios," ICLR 2025.
        - Code: https://github.com/huaqlili/unsupervised-light-enhance-ICLR2025
    """
    
    arch     : str          = "li2025"
    name     : str          = "li2025"
    tasks    : list[Task]   = [Task.LLE]
    mltypes  : list[MLType] = [MLType.UNSUPERVISED]
    model_dir: pathlib.Path = current_dir
    zoo      : dict         = box.Box()
