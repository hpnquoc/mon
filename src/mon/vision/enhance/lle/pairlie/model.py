#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements PairLIE model for low-light image enhancement.

References:
    - Paper: "Learning a Simple Low-light Image Enhancer from Paired Low-light
      Instances," CVPR 2023.
    - Code: https://github.com/zhenqifu/PairLIE
"""

__all__ = [
    "PairLIE",
]

import box

from mon.constants import MODELS
from mon.core import MLType, ModelMixin, Path, Task
from .src.net.net import net

current_file = Path(__file__).absolute()
current_dir  = current_file.parents[0]


@MODELS.register(name="pairlie", arch="pairlie")
class PairLIE(net, ModelMixin):
    """PairLIE model for low-light image enhancement.
    
    References:
        - Paper: "Learning a Simple Low-light Image Enhancer from Paired Low-light
          Instances," CVPR 2023.
        - Code: https://github.com/zhenqifu/PairLIE
    """
    
    arch     : str          = "pairlie"
    name     : str          = "pairlie"
    tasks    : list[Task]   = [Task.LLE]
    mltypes  : list[MLType] = [MLType.UNSUPERVISED]
    model_dir: Path         = current_dir
    zoo      : dict         = box.Box()
