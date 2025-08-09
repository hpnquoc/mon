#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""DAAC model for depth estimation.

References:
    - Paper: "Depth Anything At Any Condition," arXiv 2025.
    - https://github.com/HVision-NKU/DepthAnythingAC
"""

__all__ = [
    "DAAC",
    "DAV2_ViTS",
]

import os
import sys
from typing import Any

import torch

import mon.nn as nn
from mon.constants import MLType, MODELS, Task, ZOO_DIR
from mon.core import pathlib

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from depth_anything.dpt import DepthAnything_AC

current_file = pathlib.Path(__file__).absolute()
current_dir  = current_file.parents[0]


class DAAC(DepthAnything_AC, nn.ModelMixin):
    """DAAC model for depth estimation.
    
    References:
        - Paper: "Depth Anything At Any Condition," arXiv 2025.
        - https://github.com/HVision-NKU/DepthAnythingAC
    """
    
    arch     : str          = "daac"
    name     : str          = "daac"
    tasks    : list[Task]   = [Task.DEPTH]
    mltypes  : list[MLType] = [MLType.ZERO_SHOT]
    model_dir: pathlib.Path = current_dir
    zoo      : dict         = {}


@MODELS.register(name="daac_vits", arch="daac")
class DAV2_ViTS(DAAC):
    
    name: str  = "daac_vits"
    zoo : dict = {
        "pretrained": {
            "path": ZOO_DIR / "vision/dtype/depth/daac/daac_vits/pretrained/daac_vits.pth",
        },
    }
    
    def __init__(self, weights: Any = "pretrained"):
        super().__init__(
            config = {
                "encoder"        : "vits",
                "features"       : 64,
                "out_channels"   : [48, 96, 192, 384],
                "dino_pretrained": ZOO_DIR / "vision/types/depth/daac/daac_vits/pretrained/dinov2_vits14_pretrain.pth",
                "version"        : "v2",
            }
        )
        self.load_state_dict(torch.load(str(weights), weights_only=True))
