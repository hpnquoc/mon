#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Retinexformer model for low-light image enhancement.

References:
    - Paper: "Retinexformer: One-stage Retinex-based Transformer for Low-light
      Image Enhancement," ICCV 2023.
    - Code: https://github.com/caiyuanhao1998/Retinexformer
"""

__all__ = [
    "Retinexformer",
]

import box

import mon.nn as nn
from mon.constants import MLType, MODELS, Task
from mon.core import pathlib
from .src.basicsr.models.image_restoration_model import ImageCleanModel

current_file = pathlib.Path(__file__).absolute()
current_dir  = current_file.parents[0]


@MODELS.register(name="retinexformer", arch="retinexformer")
class Retinexformer(ImageCleanModel, nn.ModelMixin):
    """Retinexformer model for low-light image enhancement.
    
    References:
        - Paper: "Retinexformer: One-stage Retinex-based Transformer for Low-light
          Image Enhancement," ICCV 2023.
        - Code: https://github.com/caiyuanhao1998/Retinexformer
    """
    
    arch     : str          = "retinexformer"
    name     : str          = "retinexformer"
    tasks    : list[Task]   = [Task.LLE]
    mltypes  : list[MLType] = [MLType.SUPERVISED]
    model_dir: pathlib.Path = current_dir
    zoo      : dict         = box.Box()
