#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""FourierDiff model for zero-shot joint low-light enhancement and deblurring.

References:
    - Paper: "Fourier Priors-Guided Diffusion for Zero-Shot Joint Low-Light
      Enhancement and Deblurring," CVPR 2024.
    - Code: https://github.com/aipixel/FourierDiff
"""

__all__ = [
    "FourierDiff",
]

import os
import sys

import box

import mon.nn as nn
from mon.constants import MLType, MODELS, Task
from mon.core import pathlib

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from fourierdiff.guided_diffusion.diffusion_llie_modified import Diffusion

current_file = pathlib.Path(__file__).absolute()
current_dir  = current_file.parents[0]


@MODELS.register(name="fourierdiff", arch="fourierdiff")
class FourierDiff(Diffusion, nn.ModelMixin):
    """FourierDiff model for zero-shot joint low-light enhancement and deblurring.
    
    References:
        - Paper: "Fourier Priors-Guided Diffusion for Zero-Shot Joint Low-Light
          Enhancement and Deblurring," CVPR 2024.
        - Code: https://github.com/aipixel/FourierDiff
    """
    
    arch     : str          = "fourierdiff"
    name     : str          = "fourierdiff"
    tasks    : list[Task]   = [Task.LLE, Task.DEBLUR]
    mltypes  : list[MLType] = [MLType.ZERO_SHOT]
    model_dir: pathlib.Path = current_dir
    zoo      : dict         = box.Box()
