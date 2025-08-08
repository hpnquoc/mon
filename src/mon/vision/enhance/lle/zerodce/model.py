#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements the paper: "Zero-Reference Deep Curve Estimation for Low-Light Image
Enhancement," CVPR 2020.

References:
    - https://github.com/Li-Chongyi/Zero-DCE
"""

__all__ = [
    "ZeroDCE",
]

import box
import torch

import mon.nn as nn
from mon.constants import MLType, MODELS, Task
from mon.core import pathlib
from mon.nn import functional as F

current_file = pathlib.Path(__file__).absolute()
current_dir  = current_file.parents[0]


@MODELS.register(name="zerodce", arch="zerodce")
class ZeroDCE(nn.Module, nn.ModelMixin):
    """Implements the paper: "Zero-Reference Deep Curve Estimation for Low-Light
    Image Enhancement," CVPR 2020.
    
    References:
        - https://github.com/Li-Chongyi/Zero-DCE
    """
    
    arch     : str          = "zerodce"
    name     : str          = "zerodce"
    tasks    : list[Task]   = [Task.LLE]
    mltypes  : list[MLType] = [MLType.UNSUPERVISED]
    model_dir: pathlib.Path = current_dir
    zoo      : dict         = box.Box()

    def __init__(self):
        super().__init__()
        in_channels   = 3
        hidden_dim    = 32
        out_channels  = 24
        self.e_conv1  = nn.Conv2d(in_channels,    hidden_dim,   3, 1, 1, bias=True)
        self.e_conv2  = nn.Conv2d(hidden_dim,     hidden_dim,   3, 1, 1, bias=True)
        self.e_conv3  = nn.Conv2d(hidden_dim,     hidden_dim,   3, 1, 1, bias=True)
        self.e_conv4  = nn.Conv2d(hidden_dim,     hidden_dim,   3, 1, 1, bias=True)
        self.e_conv5  = nn.Conv2d(hidden_dim * 2, hidden_dim,   3, 1, 1, bias=True)
        self.e_conv6  = nn.Conv2d(hidden_dim * 2, hidden_dim,   3, 1, 1, bias=True)
        self.e_conv7  = nn.Conv2d(hidden_dim * 2, out_channels, 3, 1, 1, bias=True)
        self.relu     = nn.ReLU(inplace=True)
        self.maxpool  = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x1  = self.relu(self.e_conv1(x))
        x2  = self.relu(self.e_conv2(x1))
        x3  = self.relu(self.e_conv3(x2))
        x4  = self.relu(self.e_conv4(x3))
        x5  = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6  = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
        r   =    F.tanh(self.e_conv7(torch.cat([x1, x6], 1)))
        
        r1, r2, r3, r4, r5, r6, r7, r8 = torch.split(r, 3, dim=1)
        y   = x
        y   = y  + r1 * (torch.pow(y,  2) - y)
        y   = y  + r2 * (torch.pow(y,  2) - y)
        y   = y  + r3 * (torch.pow(y,  2) - y)
        y1  = y  + r4 * (torch.pow(y,  2) - y)
        y   = y1 + r5 * (torch.pow(y1, 2) - y1)
        y   = y  + r6 * (torch.pow(y,  2) - y)
        y   = y  + r7 * (torch.pow(y,  2) - y)
        y2  = y  + r8 * (torch.pow(y,  2) - y)
        
        return y2, r
