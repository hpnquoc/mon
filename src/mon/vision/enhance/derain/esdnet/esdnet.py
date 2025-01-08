#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""ESDNet.

This module implements the paper: "Learning A Spiking Neural Network for
Efficient Image Deraining," IJCAI 2024.

References:
    https://github.com/MingTian99/ESDNet
"""

from __future__ import annotations

__all__ = [

]

from typing import Any, Literal

import torch
import torch.nn.functional as F
import torch.nn.init as init

from mon import core, nn
from mon.globals import MODELS, Scheme, Task
from mon.vision.enhance import base
from spikingjelly.activation_based import functional, layer, neuron

console      = core.console
current_file = core.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# region Module

class FeatureRefinementBlock(nn.Module):
    
    def __init__(self, channel: int, reduction: int):
        super().__init__()
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.sa = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 1, 1),
            nn.Conv2d(channel, channel // 8, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, channel, 3, 1, 1),
            nn.Sigmoid()
        )
	
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.ca(x)
        t = self.sa(x)
        s = torch.mul((1 - t), a) + torch.mul(t, x)
        return s


class OverlapPatchEmbed(nn.Module):
    
    def __init__(
	    self,
	    in_c          : int  = 3,
	    embed_dim     : int  = 32,
	    spike_mode    : str  = "lif",
	    LayerNorm_type: str  = "WithBias",
	    bias          : bool = False,
    ):
        super().__init__()
        functional.set_step_mode(self, step_mode="m")
        self.proj = layer.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)
	
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x


class SpikingResidualBlock(nn.Module):
    
    def __init__(self, dim: int):
        super().__init__()
        v_th  = 0.15
        alpha = 1 / (2 ** 0.5)
        functional.set_step_mode(self, step_mode="m")
        self.residual = nn.Sequential(
            neuron.LIFNode(v_threshold=v_th, backend="cupy", step_mode="m", decay_input=False),
	        layer.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False, step_mode="m"),
	        layer.ThresholdDependentBatchNorm2d(num_features=dim, alpha=alpha, v_th=v_th, affine=True),
			
            neuron.LIFNode(v_threshold=v_th, backend="cupy", step_mode="m", decay_input=False),
	        layer.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False, step_mode="m"),
	        layer.ThresholdDependentBatchNorm2d(num_features=dim, alpha=alpha, v_th=v_th * 0.2, affine=True),
        )
        self.shortcut = nn.Sequential(
	        layer.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False, step_mode="m"),
	        layer.ThresholdDependentBatchNorm2d(num_features=dim, alpha=alpha, v_th=v_th, affine=True),
        )
        self.attn = layer.MultiDimensionalAttention(T=4, reduction_t=4, reduction_c=16, kernel_size=3, C=dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = torch.clone(x)
        out = self.residual(x) + self.shortcut(x)
        out = self.attn(out) + shortcut
        return out
    

class DownSampling(nn.Module):
    
    def __init__(self, dim: int):
        super().__init__()
        v_th  = 0.15
        alpha = 1 / (2 ** 0.5)
        functional.set_step_mode(self, step_mode="m")
        self.maxpool_conv = nn.Sequential(
	        neuron.LIFNode(v_threshold=v_th, backend="cupy", step_mode="m", decay_input=False),
	        layer.Conv2d(dim, dim * 2, kernel_size=3, stride=2, padding=1, step_mode="m", bias=False),
	        layer.ThresholdDependentBatchNorm2d(alpha=alpha, v_th=v_th, num_features=dim * 2, affine=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class UpSampling(nn.Module):
    
    def __init__(self, dim):
        super().__init__()
        v_th  = 0.15
        alpha = 1 / (2 ** 0.5)
        self.scale_factor = 2
        self.up = nn.Sequential(
            neuron.LIFNode(v_threshold=v_th, backend="cupy", step_mode="m", decay_input=False),
	        layer.Conv2d(dim, dim // 2, kernel_size=3, stride=1, padding=1, step_mode="m", bias=False),
	        layer.ThresholdDependentBatchNorm2d(alpha=alpha, v_th=v_th, num_features=dim // 2, affine=True),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        temp   = torch.zeros((input.shape[0], input.shape[1], input.shape[2], input.shape[3] * self.scale_factor, input.shape[4] * self.scale_factor)).cuda()
        output = []
        for i in range(input.shape[0]):
            # temp[i] = self.up(input[i])
            # print(input[i].shape)
            temp[i] = F.interpolate(input[i], scale_factor=self.scale_factor, mode="bilinear")
            # print(temp.shape)
            output.append(temp[i])
        out = torch.stack(output, dim=0)
        return self.up(out)
    
# endregion


# region Model

@MODELS.register(name="esdnet", arch="esdnet")
class ESDNet(base.ImageEnhancementModel):
    """Learning A Spiking Neural Network for Efficient Image Deraining.
    
    References:
        https://github.com/MingTian99/ESDNet
    """
    
    model_dir: core.Path    = current_dir
    arch     : str          = "esdnet"
    tasks    : list[Task]   = [Task.DERAIN, Task.LLIE]
    schemes  : list[Scheme] = [Scheme.SUPERVISED]
    zoo      : dict         = {}

    def __init__(
        self,
        in_channels : int = 1,
        out_channels: int = 3,
        filters     : int = 32,
        weights     : Any = None,
        *args, **kwargs
    ):
        super().__init__(
            name        = "esdnet",
            in_channels = in_channels,
            weights     = weights,
            *args, **kwargs
        )
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.filters      = filters
        
        # Construct model
        self.process_y   = nn.Sequential(
            nn.Conv2d(self.in_channels, self.filters, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.process_cb  = nn.Sequential(
            nn.Conv2d(self.in_channels, self.filters, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.process_cr  = nn.Sequential(
            nn.Conv2d(self.in_channels, self.filters, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.denoiser_cb = Denoiser(self.filters // 2)
        self.denoiser_cr = Denoiser(self.filters // 2)
        self.lum_pool    = nn.MaxPool2d(8)
        self.lum_mhsa    = MultiHeadSelfAttention(embed_size=self.filters, num_heads=4)
        self.lum_up      = nn.Upsample(scale_factor=8, mode="nearest")
        self.lum_conv    = nn.Conv2d(self.filters,     self.filters, kernel_size=1, padding=0)
        self.ref_conv    = nn.Conv2d(self.filters * 2, self.filters, kernel_size=1, padding=0)
        self.msef        = MSEFBlock(self.filters)
        self.recombine   = nn.Conv2d(self.filters * 2, self.filters, kernel_size=3, padding=1)
        
        self.final_adjustments = nn.Conv2d(self.filters, self.out_channels, kernel_size=3, padding=1)
        
        # Loss
        self.loss = Loss(reduction="mean")
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)

    def init_weights(self, m: nn.Module):
        pass
    
    def forward(self, datapoint: dict, *args, **kwargs) -> dict:
        self.assert_datapoint(datapoint)
        x         = datapoint.get("image")
        ycbcr     = self.rgb_to_ycbcr(x)
        y, cb, cr = torch.split(ycbcr, 1, dim=1)
        cb        = self.denoiser_cb(cb) + cb
        cr        = self.denoiser_cr(cr) + cr
        
        y_processed  = self.process_y(y)
        cb_processed = self.process_cb(cb)
        cr_processed = self.process_cr(cr)
        
        ref   = torch.cat([cb_processed, cr_processed], dim=1)
        lum   = y_processed
        lum_1 = self.lum_pool(lum)
        lum_1 = self.lum_mhsa(lum_1)
        lum_1 = self.lum_up(lum_1)
        lum   = lum + lum_1
        
        ref      = self.ref_conv(ref)
        shortcut = ref
        ref      = ref + 0.2 * self.lum_conv(lum)
        ref      = self.msef(ref)
        ref      = ref + shortcut
        
        recombined = self.recombine(torch.cat([ref, lum], dim=1))
        enhanced   = self.final_adjustments(recombined)
        enhanced    = torch.sigmoid(enhanced)
        return {
            "enhanced": enhanced
        }
    
# endregion
