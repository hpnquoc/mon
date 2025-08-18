#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements ... model for ...

References:
    - Paper: " ," arXiv 2025.
    - Code:
"""

__all__ = [

]

import os
import sys

import box
import torch.nn as nn

from mon.constants import MODELS, ZOO_DIR
from mon.core import MLType, ModelMixin, Path, Task, console

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
# from .src import Network

current_file = Path(__file__).absolute()
current_dir  = current_file.parents[0]


@MODELS.register(name="", arch="")
class AModel(nn.Module, ModelMixin):
    """... model for ...
    
    References:
        - Paper: " ," arXiv 2025.
        - Code:
    """
    
    arch     : str          = ""
    name     : str          = ""
    tasks    : list[Task]   = [Task.LLE]
    mltypes  : list[MLType] = [MLType.UNSUPERVISED]
    model_dir: Path         = current_dir
    zoo      : dict         = box.Box()
