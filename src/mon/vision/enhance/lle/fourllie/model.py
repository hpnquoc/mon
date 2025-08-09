#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""FourLLIE model for low-light image enhancement.

References:
    - Paper: "FourLLIE: Boosting Low-Light Image Enhancement by Fourier Frequency
      Information," ACMMM 2023.
    - Code: https://github.com/wangchx67/FourLLIE
"""

__all__ = [
    "FourLLIE",
]

import os
import sys

import box
import torch
from thop import profile

import mon.nn as nn
from mon.constants import MLType, MODELS, Task
from mon.core import pathlib
from mon.vision import types

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from fourllie.models.enhancement_model import enhancement_model

current_file = pathlib.Path(__file__).absolute()
current_dir  = current_file.parents[0]


@MODELS.register(name="fourllie", arch="fourllie")
class FourLLIE(enhancement_model, nn.ModelMixin):
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
    model_dir: pathlib.Path = current_dir
    zoo      : dict         = box.Box()
    
    def forward(self):
        self.test()
        
    def compute_efficiency_score(self, imgsz: int = 512, channels: int = 3) -> tuple[float, float]:
        h, w  = types.image_size(imgsz)
        input = torch.rand(1, channels, h, w).to(self.device)
        data  = {
            "idx": 0,
            "LQs": input,
            "nf" : input,
        }
        self.feed_data(data, need_GT=False)
        flops, params = profile(self, inputs=(), verbose=False)
        return flops, params
