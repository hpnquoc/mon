#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""SNR model for low-light image enhancement.

References:
    - Paper: "SNR-aware Low-Light Image Enhancement," CVPR 2022.
    - Code: https://github.com/dvlab-research/SNR-Aware-Low-Light-Enhance
"""

__all__ = [
    "SNR",
]

import os
import sys

import box

import mon.nn as nn
from mon.constants import MLType, MODELS, Task
from mon.core import pathlib

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from snr.models.Video_base_model4_m import VideoBaseModel

current_file = pathlib.Path(__file__).absolute()
current_dir  = current_file.parents[0]


@MODELS.register(name="snr", arch="snr")
class SNR(VideoBaseModel, nn.ModelMixin):
    """SNR model for low-light image enhancement.
    
    References:
        - Paper: "SNR-aware Low-Light Image Enhancement," CVPR 2022.
        - Code: https://github.com/dvlab-research/SNR-Aware-Low-Light-Enhance
    """
    
    arch     : str          = "snr"
    name     : str          = "snr"
    tasks    : list[Task]   = [Task.LLE]
    mltypes  : list[MLType] = [MLType.SUPERVISED]
    model_dir: pathlib.Path = current_dir
    zoo      : dict         = box.Box()
