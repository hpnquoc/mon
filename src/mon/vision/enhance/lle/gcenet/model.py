#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements GCE-Net model for low-light image enhancement."""

__all__ = [
    "GCENet",
]

import torch

from mon.constants import MODELS
from mon.core import image as I, MLType, ModelMixin, nn, Path, Task
from mon.core.nn import functional as F

current_file = Path(__file__).absolute()
current_dir  = current_file.parents[0]


# ----- Modules -----
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
        
        
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
        #
        self.depth_conv.apply(weights_init)
        self.point_conv.apply(weights_init)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        y = self.depth_conv(input)
        y = self.point_conv(y)
        return y
    
    
class ConvBlock(nn.Module):
    
    def __init__(self, in_channels: int, out_channels: int, norm: nn.Module = nn.AdaptiveBatchNorm2d):
        super().__init__()
        self.conv = DSConv(in_channels, out_channels)
        if norm:
            self.norm = norm(out_channels)
        else:
            self.norm = nn.Identity()
       
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self.norm(self.conv(input))


# ----- Model -----
@MODELS.register(name="gcenet", arch="gcenet")
class GCENet(nn.Module, ModelMixin):
    """GCE-Net model for low-light image enhancement."""
    
    arch     : str          = "gcenet"
    name     : str          = "gcenet"
    tasks    : list[Task]   = [Task.LLE]
    mltypes  : list[MLType] = [MLType.UNSUPERVISED]
    model_dir: Path         = current_dir
    zoo      : dict         = {}
    
    def __init__(self, iters: int = 8, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.iters   = iters
        in_channels  = 3
        hidden_dim   = 32
        out_channels = 3
        norm         = nn.AdaptiveBatchNorm2d
        self.e_conv1 = ConvBlock(in_channels,    hidden_dim,   norm=norm)
        self.e_conv2 = ConvBlock(hidden_dim,     hidden_dim,   norm=norm)
        self.e_conv3 = ConvBlock(hidden_dim,     hidden_dim,   norm=norm)
        self.e_conv4 = ConvBlock(hidden_dim,     hidden_dim,   norm=norm)
        self.e_conv5 = ConvBlock(hidden_dim * 2, hidden_dim,   norm=norm)
        self.e_conv6 = ConvBlock(hidden_dim * 2, hidden_dim,   norm=norm)
        self.e_conv7 = ConvBlock(hidden_dim * 2, out_channels, norm=norm)
        self.relu    = nn.LeakyReLU(inplace=False)
        self.gf      = I.GuidedFilter(kernel_size=7)
        self.bam     = I.BrightnessAttentionMap(gamma=2.6, kernel_size=9)
    
    def forward(self, image: torch.Tensor, debug: bool = False) -> tuple[torch.Tensor, ...]:
        x1 = self.relu(self.e_conv1(image))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
        r  =    F.tanh(self.e_conv7(torch.cat([x1, x6], 1)))
        
        # Enhancement loop
        bam      = self.bam(image)
        b        = None
        d        = None
        enhanced = image
        for _ in range(0, self.iters):
            b = enhanced * (1 - bam)
            d = enhanced * bam
            enhanced = b + d + r * (torch.pow(d, 2) - d)
        
        # Guided Filter
        enhanced = self.gf(image, enhanced)
        
        if debug:
            return {
                "bam": bam,
                "b"  : b,
                "d"  : d,
            }, enhanced
        else:
            return enhanced
