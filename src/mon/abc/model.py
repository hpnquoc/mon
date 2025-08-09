#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""... model for ...

References:
    - Paper: " ," arXiv 2025.
    - Code:
"""

__all__ = [

]

import os
import sys
import box
import torch

import mon.nn as nn
from mon.constants import MLType, MODELS, Task
from mon.core import pathlib
from mon.nn import functional as F

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
# from lightendiffusion.models import DenoisingDiffusion, DiffusiveRestoration

current_file = pathlib.Path(__file__).absolute()
current_dir  = current_file.parents[0]


@MODELS.register(name="", arch="")
class AModel(nn.Module, nn.ModelMixin):
    """... model for ...
    
    References:
        - Paper: " ," arXiv 2025.
        - Code:
    """
    
    arch     : str          = ""
    name     : str          = ""
    tasks    : list[Task]   = [Task.LLE]
    mltypes  : list[MLType] = [MLType.UNSUPERVISED]
    model_dir: pathlib.Path = current_dir
    zoo      : dict         = box.Box()
