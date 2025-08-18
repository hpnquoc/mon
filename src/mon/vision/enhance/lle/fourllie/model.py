#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements FourLLIE model for low-light image enhancement.

References:
    - Paper: "FourLLIE: Boosting Low-Light Image Enhancement by Fourier Frequency
      Information," ACMMM 2023.
    - Code: https://github.com/wangchx67/FourLLIE
"""

__all__ = [
    "FourLLIE",
    "option",
    "read_img",
    "tensor2img",
]

import box
import torch
from thop import profile

from mon.constants import MODELS
from mon.core import image as I, MLType, ModelMixin, Path, Task
from .src.data.util import read_img
from .src.models.enhancement_model import enhancement_model
from .src.option import options as option
from .src.utils.util import tensor2img

current_file = Path(__file__).absolute()
current_dir  = current_file.parents[0]


@MODELS.register(name="fourllie", arch="fourllie")
class FourLLIE(enhancement_model, ModelMixin):
    """FourLLIE model for low-light image enhancement.
    
    References:
        - Paper: "FourLLIE: Boosting Low-Light Image Enhancement by Fourier Frequency
          Information," ACMMM 2023.
        - Code: https://github.com/wangchx67/FourLLIE
    """
    
    arch     : str          = "fourllie"
    name     : str          = "fourllie"
    tasks    : list[Task]   = [Task.LLE]
    mltypes  : list[MLType] = [MLType.SUPERVISED]
    model_dir: Path         = current_dir
    zoo      : dict         = box.Box()
    
    def forward(self):
        self.test()
        
    def compute_complexity(self, imgsz: int = 512, channels: int = 3) -> tuple[float, float]:
        h, w  = I.imgsz(imgsz)
        input = torch.rand(1, channels, h, w).to(self.device)
        data  = {
            "idx": 0,
            "LQs": input,
            "nf" : input,
        }
        self.feed_data(data, need_GT=False)
        flops, params = profile(self, inputs=(), verbose=False)
        return flops, params
