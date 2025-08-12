#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""NeurOP model for image retouching.

References:
    - Paper: "Neural Color Operators for Sequential Image Retouching," ECCV 2022.
    - Code: https://github.com/amberwangyili/neurop
"""

__all__ = [
    "NeurOP",
    "NeurOPInit",
]

import box

import mon.nn as nn
from mon.constants import MLType, MODELS, Task
from mon.core import pathlib
from .src.models.model import FinetuneModel, InitModel

current_file = pathlib.Path(__file__).absolute()
current_dir  = current_file.parents[0]


@MODELS.register(name="neurop", arch="neurop")
class NeurOP(FinetuneModel, nn.ModelMixin):
    """NeurOP model for image retouching.
    
    References:
        - Paper: "Neural Color Operators for Sequential Image Retouching," ECCV 2022.
        - Code: https://github.com/amberwangyili/neurop
    """
    
    arch     : str          = "neurop"
    name     : str          = "neurop"
    tasks    : list[Task]   = [Task.RETOUCH]
    mltypes  : list[MLType] = [MLType.SUPERVISED]
    model_dir: pathlib.Path = current_dir
    zoo      : dict         = box.Box()


class NeurOPInit(InitModel):
    """NeurOP model for image retouching.
    
    References:
        - Paper: "Neural Color Operators for Sequential Image Retouching," ECCV 2022.
        - Code: https://github.com/amberwangyili/neurop
    """
    pass
