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

from typing import Any

import box

from mon.constants import MODELS, ROOT_DIR
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
    zoo      : dict         = box.Box({
        "lolv1"    : {
            "url"        : None,
            "path"       : ROOT_DIR / "zoo/vision/enhance/lle/li2025/li2025/lolv1/li2025_lolv1.pth",
            "num_classes": None,
        },
        "lolv2real": {
            "url"        : None,
            "path"       : ROOT_DIR / "zoo/vision/enhance/lle/li2025/li2025/lolv2real/li2025_lolv2real.pth",
            "num_classes": None,
        },
        "sice"     : {
            "url"        : None,
            "path"       : ROOT_DIR / "zoo/vision/enhance/lle/li2025/li2025/sice/li2025_sice.pth",
            "num_classes": None,
        },
    })
    
    def __init__(self, weights: Any = None):
        super().__init__()
        # Load weights
        self.load_weights(weights)
