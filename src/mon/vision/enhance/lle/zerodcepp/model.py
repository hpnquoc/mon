#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements the paper: "Zero-Reference Deep Curve Estimation for Low-Light Image
Enhancement," CVPR 2020.

References:
    - https://github.com/Li-Chongyi/Zero-DCE
"""

__all__ = [
    "ZeroDCEpp",
]

import box
import torch

import mon.nn as nn
from mon.constants import MLType, MODELS, Task
from mon.core import pathlib
from mon.nn import functional as F

current_file = pathlib.Path(__file__).absolute()
current_dir  = current_file.parents[0]


class DSConv(nn.Module):
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.depth_conv = nn.Conv2d(
            in_channels  = in_channels,
            out_channels = in_channels,
            kernel_size  = 3,
            stride       = 1,
            padding      = 1,
            groups       = in_channels
        )
        self.point_conv = nn.Conv2d(
            in_channels  = in_channels,
            out_channels = out_channels,
            kernel_size  = 1,
            stride       = 1,
            padding      = 0,
            groups       = 1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.depth_conv(x)
        y = self.point_conv(y)
        return y


@MODELS.register(name="zerodce++", arch="zerodce++")
class ZeroDCEpp(nn.Module):
    """Implements the paper: "Zero-Reference Deep Curve Estimation for Low-Light Image
    Enhancement," CVPR 2020.
    
    References:
        - https://github.com/Li-Chongyi/Zero-DCE
    """
    
    arch     : str          = "zerodce++"
    name     : str          = "zerodce++"
    tasks    : list[Task]   = [Task.LLE]
    mltypes  : list[MLType] = [MLType.UNSUPERVISED]
    model_dir: pathlib.Path = current_dir
    zoo      : dict         = box.Box()
    
    def __init__(self, scale_factor: float = 1.0):
        super().__init__()
        self.scale_factor = scale_factor
        
        in_channels   = 3
        hidden_dim    = 32
        out_channels  = 3
        self.e_conv1  = DSConv(in_channels, hidden_dim)
        self.e_conv2  = DSConv(hidden_dim, hidden_dim)
        self.e_conv3  = DSConv(hidden_dim, hidden_dim)
        self.e_conv4  = DSConv(hidden_dim, hidden_dim)
        self.e_conv5  = DSConv(hidden_dim * 2, hidden_dim)
        self.e_conv6  = DSConv(hidden_dim * 2, hidden_dim)
        self.e_conv7  = DSConv(hidden_dim * 2, out_channels)
        self.relu     = nn.ReLU(inplace=True)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=self.scale_factor)
        
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.scale_factor == 1:
            x_down = x
        else:
            x_down = F.interpolate(x, scale_factor=1 / self.scale_factor, mode="bilinear")

        x1 = self.relu(self.e_conv1(x_down))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
        r  =    F.tanh(self.e_conv7(torch.cat([x1, x6], 1)))
        
        if self.scale_factor == 1:
            r = r
        else:
            r = self.upsample(r)
            
        y = self.enhance(x, r)
        return y, r
    
    def enhance(self, x: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        y = x
        y = y + r * (torch.pow(y, 2) - y)
        y = y + r * (torch.pow(y, 2) - y)
        y = y + r * (torch.pow(y, 2) - y)
        y = y + r * (torch.pow(y, 2) - y)
        y = y + r * (torch.pow(y, 2) - y)
        y = y + r * (torch.pow(y, 2) - y)
        y = y + r * (torch.pow(y, 2) - y)
        y = y + r * (torch.pow(y, 2) - y)
        return y
