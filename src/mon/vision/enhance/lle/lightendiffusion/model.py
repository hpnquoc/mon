#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""LightenDiffusion model for low-light image enhancement.

References:
    - Paper: "LightenDiffusion: Unsupervised Low-Light Image Enhancement with
      Latent-Retinex Diffusion Models," ECCV 2024.
    - Code: https://github.com/JianghaiSCU/LightenDiffusion
"""

__all__ = [
    "LightenDiffusion",
]

import os
import sys

import box

import mon.nn as nn
from mon.constants import MLType, MODELS, Task
from mon.core import pathlib

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from lightendiffusion.models import DenoisingDiffusion

current_file = pathlib.Path(__file__).absolute()
current_dir  = current_file.parents[0]


@MODELS.register(name="lightendiffusion", arch="lightendiffusion")
class LightenDiffusion(DenoisingDiffusion, nn.ModelMixin):
    """LightenDiffusion model for low-light image enhancement.
    
    References:
        - Paper: "LightenDiffusion: Unsupervised Low-Light Image Enhancement with
          Latent-Retinex Diffusion Models," ECCV 2024.
        - Code: https://github.com/JianghaiSCU/LightenDiffusion
    """
    
    arch     : str          = "lightendiffusion"
    name     : str          = "lightendiffusion"
    tasks    : list[Task]   = [Task.LLE]
    mltypes  : list[MLType] = [MLType.SUPERVISED]
    model_dir: pathlib.Path = current_dir
    zoo      : dict         = box.Box()
