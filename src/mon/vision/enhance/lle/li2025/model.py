#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements Li2025 model for low-light image enhancement.

References:
    - Paper: "Interpretable Unsupervised Joint Denoising and Enhancement for
      Real-World low-light Scenarios," ICLR 2025.
    - Code: https://github.com/huaqlili/unsupervised-light-enhance-ICLR2025
"""

__all__ = [
    "Li2025",
]

import box

from mon.constants import MODELS
from mon.core import MLType, ModelMixin, Path, Task
from .src.net.lformer import net

current_file = Path(__file__).absolute()
current_dir  = current_file.parents[0]


@MODELS.register(name="li2025", arch="li2025")
class Li2025(net, ModelMixin):
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
    model_dir: Path         = current_dir
    zoo      : dict         = box.Box()
