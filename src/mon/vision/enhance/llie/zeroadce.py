#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements Zero-ADCE models.
"""

from __future__ import annotations

__all__ = [
    "ZeroADCE",
]

from typing import Any

import kornia
import torch

import mon
from mon.globals import MODELS, ModelPhase
from mon.vision import core, nn, prior
from mon.vision.enhance.llie import base
from mon.vision.nn import functional as F

math         = core.math
console      = core.console
_current_dir = core.Path(__file__).absolute().parent


# region Loss

class ZeroReferenceLoss(nn.Loss):
    
    def __init__(
        self,
        bri_gamma      : float = 2.8,
        exp_patch_size : int   = 16,
        exp_mean_val   : float = 0.6,
        spa_num_regions: int   = 4,     # 4
        spa_patch_size : int   = 4,     # 4
        weight_bri     : float = 1,
        weight_col     : float = 5,
        weight_crl     : float = 1,     # 20
        weight_edge    : float = 5,
        weight_exp     : float = 10,
        weight_kl      : float = 5,     # 5
        weight_spa     : float = 1,
        weight_tvA     : float = 1600,  # 200
        reduction      : str   = "mean",
        verbose        : bool  = False,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.weight_bri  = weight_bri
        self.weight_col  = weight_col
        self.weight_crl  = weight_crl
        self.weight_edge = weight_edge
        self.weight_exp  = weight_exp
        self.weight_kl   = weight_kl
        self.weight_spa  = weight_spa
        self.weight_tvA  = weight_tvA
        self.verbose     = verbose
        
        self.loss_bri  = nn.BrightnessConstancyLoss(reduction=reduction, gamma=bri_gamma)
        self.loss_col  = nn.ColorConstancyLoss(reduction=reduction)
        self.loss_crl  = nn.ChannelRatioConsistencyLoss(reduction=reduction)
        self.loss_kl   = nn.ChannelConsistencyLoss(reduction=reduction)
        self.loss_edge = nn.EdgeConstancyLoss(reduction=reduction)
        self.loss_exp  = nn.ExposureControlLoss(
            reduction  = reduction,
            patch_size = exp_patch_size,
            mean_val   = exp_mean_val,
        )
        self.loss_spa  = nn.SpatialConsistencyLoss(
            num_regions = spa_num_regions,
            patch_size  = spa_patch_size,
            reduction   = reduction,
        )
        self.loss_tvA  = nn.IlluminationSmoothnessLoss(reduction=reduction)
    
    def __str__(self) -> str:
        return f"zero-reference loss"
    
    def forward(
        self,
        input   : torch.Tensor | list[torch.Tensor],
        target  : list[torch.Tensor],
        previous: torch.Tensor = None,
        **_
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(target, list | tuple):
            if len(target) == 2:
                a       = target[-2]
                enhance = target[-1]
            elif len(target) == 3:
                a       = target[-3]
                p       = target[-2]
                enhance = target[-1]
        else:
            raise TypeError
        loss_bri  = self.loss_bri(input=p, target=input)              if self.weight_bri  > 0 else 0
        loss_col  = self.loss_col(input=enhance)                      if self.weight_col  > 0 else 0
        loss_edge = self.loss_edge(input=enhance, target=input)       if self.weight_edge > 0 else 0
        loss_exp  = self.loss_exp(input=enhance)                      if self.weight_exp  > 0 else 0
        loss_kl   = self.loss_kl(input=enhance, target=input)         if self.weight_kl   > 0 else 0
        loss_spa  = self.loss_spa(input=enhance, target=input)        if self.weight_spa  > 0 else 0
        loss_tvA  = self.loss_tvA(input=a)                            if self.weight_tvA  > 0 else 0
        if previous is not None and (enhance.shape == previous.shape):
            loss_crl = self.loss_crl(input=enhance, target=previous)  if self.weight_crl  > 0 else 0
        else:                                                                             
            loss_crl = self.loss_crl(input=enhance, target=input)     if self.weight_crl  > 0 else 0
        
        loss = (
             self.weight_bri   * loss_bri
            + self.weight_col  * loss_col
            + self.weight_crl  * loss_crl
            + self.weight_edge * loss_edge
            + self.weight_exp  * loss_exp
            + self.weight_tvA  * loss_tvA
            + self.weight_kl   * loss_kl
            + self.weight_spa  * loss_spa
        )
        
        if self.verbose:
            console.log(f"{self.loss_bri.__str__():<30} : {loss_bri}")
            console.log(f"{self.loss_col.__str__():<30} : {loss_col}")
            console.log(f"{self.loss_edge.__str__():<30}: {loss_edge}")
            console.log(f"{self.loss_exp.__str__():<30} : {loss_exp}")
            console.log(f"{self.loss_kl.__str__():<30}  : {loss_kl}")
            console.log(f"{self.loss_spa.__str__():<30} : {loss_spa}")
            console.log(f"{self.loss_tvA.__str__():<30} : {loss_tvA}")
        return loss, enhance
        
# endregion


# region Model

@MODELS.register(name="zeroadce")
@MODELS.register(name="zero-adce")
class ZeroADCE(base.LowLightImageEnhancementModel):
    """Zero-ADCE (Zero-Reference Attention Deep Curve Estimation) model.
    
    See Also: :class:`mon.vision.enhance.llie.base.LowLightImageEnhancementModel`
    """
    
    configs     = {}
    zoo         = {}
    map_weights = {}
    
    def __init__(
        self,
        config       : Any                = None,
        loss         : Any                = ZeroReferenceLoss(),
        variant      :         str | None = None,
        num_channels : int   | str        = 32,
        scale_factor : float | str        = 1.0,
        gamma        : float | str | None = 2.8,
        num_iters    : int   | str        = 8,
        unsharp_sigma: int   | str | None = None,
        *args, **kwargs
    ):
        super().__init__(
            config = config,
            loss   = loss,
            *args, **kwargs
        )
        variant            = mon.to_int(variant)
        self.variant       = f"{variant:04d}" if isinstance(variant, int) else None
        self.num_channels  = mon.to_int(num_channels)    or 32
        self.scale_factor  = mon.to_float(scale_factor)  or 1.0
        self.gamma         = mon.to_float(gamma)         or 2.8
        self.num_iters     = mon.to_int(num_iters)       or 8
        self.unsharp_sigma = mon.to_float(unsharp_sigma) or None
        self.previous      = None

        if variant is None:  # Default model
            self.out_channels = 3
            self.conv1        = nn.ABSConv2dS(self.channels,         self.num_channels, 3, 1, 1, bias=True, norm2=nn.HalfInstanceNorm2d)
            self.conv2        = nn.ABSConv2dS(self.num_channels,     self.num_channels, 3, 1, 1, bias=True, norm2=nn.HalfInstanceNorm2d)
            self.conv3        = nn.ABSConv2dS(self.num_channels,     self.num_channels, 3, 1, 1, bias=True, norm2=nn.HalfInstanceNorm2d)
            self.conv4        = nn.ABSConv2dS(self.num_channels,     self.num_channels, 3, 1, 1, bias=True, norm2=nn.HalfInstanceNorm2d)
            self.conv5        = nn.ABSConv2dS(self.num_channels * 2, self.num_channels, 3, 1, 1, bias=True, norm2=nn.HalfInstanceNorm2d)
            self.conv6        = nn.ABSConv2dS(self.num_channels * 2, self.num_channels, 3, 1, 1, bias=True, norm2=nn.HalfInstanceNorm2d)
            self.conv7        =     nn.Conv2d(self.num_channels * 2, self.out_channels, 3, 1, 1, bias=True)
            self.act          = nn.ReLU(inplace=True)
            self.upsample     = nn.UpsamplingBilinear2d(self.scale_factor)
            self.loss         = ZeroReferenceLoss(
                exp_patch_size  = 16,
                exp_mean_val    = 0.6,
                spa_num_regions = 8,
                spa_patch_size  = 4,
                weight_bri      = 0,
                weight_col      = 5,
                weight_crl      = 0.1,
                weight_edge     = 1,
                weight_exp      = 10,
                weight_kl       = 5,
                weight_spa      = 1,
                weight_tvA      = 1600,
                reduction       = "mean",
            )
            self.apply(self.init_weights)
        else:
            self.config_model_variant()

    def config_model_variant(self):
        """Config the model based on ``self.variant``.
        Mainly used in ablation study.
        """
        # self.num_iters     = 9
        # self.unsharp_sigma = 2.5
        self.previous      = None
        self.out_channels  = 3

        # Variant code: [aa][l][i]
        # i: inference mode
        if self.variant[3] == "0":
            self.out_channels = 3
        elif self.variant[3] == "1":
            self.gamma        = self.gamma or 2.8
            self.out_channels = 3
        elif self.variant[3] == "2":
            self.gamma        = self.gamma or 2.8
            self.out_channels = 3
        elif self.variant[3] == "3":
            self.gamma        = self.gamma or 2.8
            self.out_channels = 3
        else:
            raise ValueError

        # Variant code: [aa][l][i]
        # aa: architecture
        if self.variant[0:2] == "00":
            self.conv1    = nn.ABSConv2dS(self.channels,         self.num_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv2    = nn.ABSConv2dS(self.num_channels,     self.num_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv3    = nn.ABSConv2dS(self.num_channels,     self.num_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv4    = nn.ABSConv2dS(self.num_channels,     self.num_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv5    = nn.ABSConv2dS(self.num_channels * 2, self.num_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv6    = nn.ABSConv2dS(self.num_channels * 2, self.num_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv7    = nn.ABSConv2dS(self.num_channels * 2, self.out_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.act      = nn.ReLU(inplace=True)
            self.upsample = nn.UpsamplingBilinear2d(self.scale_factor)
            self.apply(self.init_weights)
        elif self.variant[0:2] == "01":
            self.conv1    = nn.ABSConv2dS1(self.channels,         self.num_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv2    = nn.ABSConv2dS1(self.num_channels,     self.num_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv3    = nn.ABSConv2dS1(self.num_channels,     self.num_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv4    = nn.ABSConv2dS1(self.num_channels,     self.num_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv5    = nn.ABSConv2dS1(self.num_channels * 2, self.num_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv6    = nn.ABSConv2dS1(self.num_channels * 2, self.num_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv7    = nn.ABSConv2dS1(self.num_channels * 2, self.out_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.act      = nn.ReLU(inplace=True)
            self.upsample = nn.UpsamplingBilinear2d(self.scale_factor)
            self.apply(self.init_weights)
        elif self.variant[0:2] == "02":
            self.conv1    = nn.ABSConv2dS2(self.channels,         self.num_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv2    = nn.ABSConv2dS2(self.num_channels,     self.num_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv3    = nn.ABSConv2dS2(self.num_channels,     self.num_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv4    = nn.ABSConv2dS2(self.num_channels,     self.num_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv5    = nn.ABSConv2dS2(self.num_channels * 2, self.num_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv6    = nn.ABSConv2dS2(self.num_channels * 2, self.num_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv7    = nn.ABSConv2dS2(self.num_channels * 2, self.out_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.act      = nn.ReLU(inplace=True)
            self.upsample = nn.UpsamplingBilinear2d(self.scale_factor)
            self.apply(self.init_weights)
        elif self.variant[0:2] == "03":
            self.conv1    = nn.ABSConv2dS3(self.channels,         self.num_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv2    = nn.ABSConv2dS3(self.num_channels,     self.num_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv3    = nn.ABSConv2dS3(self.num_channels,     self.num_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv4    = nn.ABSConv2dS3(self.num_channels,     self.num_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv5    = nn.ABSConv2dS3(self.num_channels * 2, self.num_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv6    = nn.ABSConv2dS3(self.num_channels * 2, self.num_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv7    = nn.ABSConv2dS3(self.num_channels * 2, self.out_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.act      = nn.ReLU(inplace=True)
            self.upsample = nn.UpsamplingBilinear2d(self.scale_factor)
            self.apply(self.init_weights)
        elif self.variant[0:2] == "04":
            self.conv1    = nn.ABSConv2dS4(self.channels,         self.num_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv2    = nn.ABSConv2dS4(self.num_channels,     self.num_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv3    = nn.ABSConv2dS4(self.num_channels,     self.num_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv4    = nn.ABSConv2dS4(self.num_channels,     self.num_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv5    = nn.ABSConv2dS4(self.num_channels * 2, self.num_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv6    = nn.ABSConv2dS4(self.num_channels * 2, self.num_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv7    = nn.ABSConv2dS4(self.num_channels * 2, self.out_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.act      = nn.ReLU(inplace=True)
            self.upsample = nn.UpsamplingBilinear2d(self.scale_factor)
            self.apply(self.init_weights)
        elif self.variant[0:2] == "05":
            self.conv1    = nn.ABSConv2dS5(self.channels,         self.num_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv2    = nn.ABSConv2dS5(self.num_channels,     self.num_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv3    = nn.ABSConv2dS5(self.num_channels,     self.num_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv4    = nn.ABSConv2dS5(self.num_channels,     self.num_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv5    = nn.ABSConv2dS5(self.num_channels * 2, self.num_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv6    = nn.ABSConv2dS5(self.num_channels * 2, self.num_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv7    = nn.ABSConv2dS5(self.num_channels * 2, self.out_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.act      = nn.ReLU(inplace=True)
            self.upsample = nn.UpsamplingBilinear2d(self.scale_factor)
            self.apply(self.init_weights)
        elif self.variant[0:2] == "06":
            self.conv1    = nn.ABSConv2dS6(self.channels,         self.num_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv2    = nn.ABSConv2dS6(self.num_channels,     self.num_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv3    = nn.ABSConv2dS6(self.num_channels,     self.num_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv4    = nn.ABSConv2dS6(self.num_channels,     self.num_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv5    = nn.ABSConv2dS6(self.num_channels * 2, self.num_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv6    = nn.ABSConv2dS6(self.num_channels * 2, self.num_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv7    = nn.ABSConv2dS6(self.num_channels * 2, self.out_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.act      = nn.ReLU(inplace=True)
            self.upsample = nn.UpsamplingBilinear2d(self.scale_factor)
            self.apply(self.init_weights)
        elif self.variant[0:2] == "07":
            self.conv1    = nn.ABSConv2dS7(self.channels,         self.num_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv2    = nn.ABSConv2dS7(self.num_channels,     self.num_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv3    = nn.ABSConv2dS7(self.num_channels,     self.num_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv4    = nn.ABSConv2dS7(self.num_channels,     self.num_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv5    = nn.ABSConv2dS7(self.num_channels * 2, self.num_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv6    = nn.ABSConv2dS7(self.num_channels * 2, self.num_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv7    = nn.ABSConv2dS7(self.num_channels * 2, self.out_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.act      = nn.ReLU(inplace=True)
            self.upsample = nn.UpsamplingBilinear2d(self.scale_factor)
            self.apply(self.init_weights)
        elif self.variant[0:2] == "08":
            self.conv1    = nn.ABSConv2dS8(self.channels,         self.num_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv2    = nn.ABSConv2dS8(self.num_channels,     self.num_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv3    = nn.ABSConv2dS8(self.num_channels,     self.num_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv4    = nn.ABSConv2dS8(self.num_channels,     self.num_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv5    = nn.ABSConv2dS8(self.num_channels * 2, self.num_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv6    = nn.ABSConv2dS8(self.num_channels * 2, self.num_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv7    = nn.ABSConv2dS8(self.num_channels * 2, self.out_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.act      = nn.ReLU(inplace=True)
            self.upsample = nn.UpsamplingBilinear2d(self.scale_factor)
            self.apply(self.init_weights)
        elif self.variant[0:2] == "09":
            self.conv1    = nn.ABSConv2dS9(self.channels,         self.num_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv2    = nn.ABSConv2dS9(self.num_channels,     self.num_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv3    = nn.ABSConv2dS9(self.num_channels,     self.num_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv4    = nn.ABSConv2dS9(self.num_channels,     self.num_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv5    = nn.ABSConv2dS9(self.num_channels * 2, self.num_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv6    = nn.ABSConv2dS9(self.num_channels * 2, self.num_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv7    = nn.ABSConv2dS9(self.num_channels * 2, self.out_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.act      = nn.ReLU(inplace=True)
            self.upsample = nn.UpsamplingBilinear2d(self.scale_factor)
            self.apply(self.init_weights)
        elif self.variant[0:2] == "10":
            self.conv1    = nn.ABSConv2dS10(self.channels,         self.num_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv2    = nn.ABSConv2dS10(self.num_channels,     self.num_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv3    = nn.ABSConv2dS10(self.num_channels,     self.num_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv4    = nn.ABSConv2dS10(self.num_channels,     self.num_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv5    = nn.ABSConv2dS10(self.num_channels * 2, self.num_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv6    = nn.ABSConv2dS10(self.num_channels * 2, self.num_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv7    = nn.ABSConv2dS10(self.num_channels * 2, self.out_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.act      = nn.ReLU(inplace=True)
            self.upsample = nn.UpsamplingBilinear2d(self.scale_factor)
            self.apply(self.init_weights)
        elif self.variant[0:2] == "11":
            self.conv1    = nn.ABSConv2dS11(self.channels,         self.num_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv2    = nn.ABSConv2dS11(self.num_channels,     self.num_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv3    = nn.ABSConv2dS11(self.num_channels,     self.num_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv4    = nn.ABSConv2dS11(self.num_channels,     self.num_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv5    = nn.ABSConv2dS11(self.num_channels * 2, self.num_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv6    = nn.ABSConv2dS11(self.num_channels * 2, self.num_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv7    = nn.ABSConv2dS11(self.num_channels * 2, self.out_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.act      = nn.ReLU(inplace=True)
            self.upsample = nn.UpsamplingBilinear2d(self.scale_factor)
            self.apply(self.init_weights)
        elif self.variant[0:2] == "12":
            self.conv1    = nn.ABSConv2dS12(self.channels,         self.num_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv2    = nn.ABSConv2dS12(self.num_channels,     self.num_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv3    = nn.ABSConv2dS12(self.num_channels,     self.num_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv4    = nn.ABSConv2dS12(self.num_channels,     self.num_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv5    = nn.ABSConv2dS12(self.num_channels * 2, self.num_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv6    = nn.ABSConv2dS12(self.num_channels * 2, self.num_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv7    = nn.ABSConv2dS12(self.num_channels * 2, self.out_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.act      = nn.ReLU(inplace=True)
            self.upsample = nn.UpsamplingBilinear2d(self.scale_factor)
            self.apply(self.init_weights)
        elif self.variant[0:2] == "13":
            self.conv1    = nn.ABSConv2dS13(self.channels,         self.num_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv2    = nn.ABSConv2dS13(self.num_channels,     self.num_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv3    = nn.ABSConv2dS13(self.num_channels,     self.num_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv4    = nn.ABSConv2dS13(self.num_channels,     self.num_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv5    = nn.ABSConv2dS13(self.num_channels * 2, self.num_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv6    = nn.ABSConv2dS13(self.num_channels * 2, self.num_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv7    = nn.ABSConv2dS13(self.num_channels * 2, self.out_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.act      = nn.ReLU(inplace=True)
            self.upsample = nn.UpsamplingBilinear2d(self.scale_factor)
            self.apply(self.init_weights)
        elif self.variant[0:2] == "14":
            self.conv1    = nn.ABSConv2dS14(self.channels,         self.num_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv2    = nn.ABSConv2dS14(self.num_channels,     self.num_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv3    = nn.ABSConv2dS14(self.num_channels,     self.num_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv4    = nn.ABSConv2dS14(self.num_channels,     self.num_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv5    = nn.ABSConv2dS14(self.num_channels * 2, self.num_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv6    = nn.ABSConv2dS14(self.num_channels * 2, self.num_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv7    = nn.ABSConv2dS14(self.num_channels * 2, self.out_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.act      = nn.ReLU(inplace=True)
            self.upsample = nn.UpsamplingBilinear2d(self.scale_factor)
            self.apply(self.init_weights)
        elif self.variant[0:2] == "15":
            self.conv1    = nn.ABSConv2dS15(self.channels,         self.num_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv2    = nn.ABSConv2dS15(self.num_channels,     self.num_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv3    = nn.ABSConv2dS15(self.num_channels,     self.num_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv4    = nn.ABSConv2dS15(self.num_channels,     self.num_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv5    = nn.ABSConv2dS15(self.num_channels * 2, self.num_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv6    = nn.ABSConv2dS15(self.num_channels * 2, self.num_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv7    = nn.ABSConv2dS15(self.num_channels * 2, self.out_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.act      = nn.ReLU(inplace=True)
            self.upsample = nn.UpsamplingBilinear2d(self.scale_factor)
            self.apply(self.init_weights)
        elif self.variant[0:2] == "16":
            self.conv1    = nn.ABSConv2dS16(self.channels,         self.num_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv2    = nn.ABSConv2dS16(self.num_channels,     self.num_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv3    = nn.ABSConv2dS16(self.num_channels,     self.num_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv4    = nn.ABSConv2dS16(self.num_channels,     self.num_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv5    = nn.ABSConv2dS16(self.num_channels * 2, self.num_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv6    = nn.ABSConv2dS16(self.num_channels * 2, self.num_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv7    = nn.ABSConv2dS16(self.num_channels * 2, self.out_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.act      = nn.ReLU(inplace=True)
            self.upsample = nn.UpsamplingBilinear2d(self.scale_factor)
            self.apply(self.init_weights)
        elif self.variant[0:2] == "17":
            self.conv1    = nn.ABSConv2dS17(self.channels,         self.num_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv2    = nn.ABSConv2dS17(self.num_channels,     self.num_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv3    = nn.ABSConv2dS17(self.num_channels,     self.num_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv4    = nn.ABSConv2dS17(self.num_channels,     self.num_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv5    = nn.ABSConv2dS17(self.num_channels * 2, self.num_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv6    = nn.ABSConv2dS17(self.num_channels * 2, self.num_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.conv7    = nn.ABSConv2dS17(self.num_channels * 2, self.out_channels, 3, 1, 1, bias=True, p=0.25, min_mid_channels=4, attn=True, norm1=nn.HalfInstanceNorm2d, norm2=nn.HalfInstanceNorm2d)
            self.act      = nn.ReLU(inplace=True)
            self.upsample = nn.UpsamplingBilinear2d(self.scale_factor)
            self.apply(self.init_weights)
        else:
            raise ValueError

        # Variant code: [aa][l][i]
        # l: loss function
        weight_tvA = 1600 if self.out_channels == 3 else 200
        if self.variant[2] == "0":  # Zero-DCE Loss
            # NOT WORKING: over-exposed artifacts, enhance noises
            self.loss = ZeroReferenceLoss(
                exp_patch_size  = 16,
                exp_mean_val    = 0.6,
                spa_num_regions = 4,
                spa_patch_size  = 4,
                weight_bri      = 0,
                weight_col      = 5,
                weight_crl      = 0,
                weight_edge     = 0,
                weight_exp      = 10,
                weight_kl       = 0,
                weight_spa      = 1,
                weight_tvA      = weight_tvA,
                reduction       = "mean",
            )
        elif self.variant[2] == "1":  # New Loss
            self.loss = ZeroReferenceLoss(
                exp_patch_size  = 16,
                exp_mean_val    = 0.6,
                spa_num_regions = 8,
                spa_patch_size  = 4,
                weight_bri      = 0,
                weight_col      = 5,
                weight_crl      = 0.1,
                weight_edge     = 1,
                weight_exp      = 10,
                weight_kl       = 0.1,
                weight_spa      = 1,
                weight_tvA      = weight_tvA,
                reduction       = "mean",
            )
        else:
            raise ValueError

    @property
    def config_dir(self) -> core.Path:
        return core.Path(__file__).absolute().parent / "config"
    
    def init_weights(self, m: nn.Module):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            if hasattr(m, "conv"):
                m.conv.weight.data.normal_(0.0, 0.02)  # 0.02
            elif hasattr(m, "dw_conv"):
                m.dw_conv.weight.data.normal_(0.0, 0.02)  # 0.02
            elif hasattr(m, "pw_conv"):
                m.pw_conv.weight.data.normal_(0.0, 0.02)  # 0.02
            elif hasattr(m, "weight"):
                m.weight.data.normal_(0.0, 0.02)  # 0.02
    
    def forward_loss(
        self,
        input : torch.Tensor,
        target: torch.Tensor | None,
        *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass with loss value. Loss function may need more arguments
        beside the ground-truth and prediction values. For calculating the
        metrics, we only need the final predictions and ground-truth.

        Args:
            input: An input of shape :math:`[N, C, H, W]`.
            target: A ground-truth of shape :math:`[N, C, H, W]`. Default: ``None``.
            
        Return:
            Predictions and loss value.
        """
        if self.variant is not None:
            pred = self.forward_once_variant(input=input, *args, **kwargs)
        else:
            pred = self.forward(input=input, *args, **kwargs)
        loss, self.previous = self.loss(input, pred, self.previous) if self.loss else (None, None)
        loss += self.regularization_loss(alpha=0.1)
        return pred[-1], loss
    
    def forward_once(
        self,
        input    : torch.Tensor,
        profile  : bool = False,
        out_index: int = -1,
        *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass once. Implement the logic for a single forward pass.

        Args:
            input: An input of shape :math:`[N, C, H, W]`.
            profile: Measure processing time. Default: ``False``.
            out_index: Return specific layer's output from :param:`out_index`.
                Default: ``-1`` means the last layer.
                
        Return:
            Predictions.
        """
        x = input
        
        # Downsampling
        x_down = x
        if self.scale_factor != 1:
            x_down = F.interpolate(x, scale_factor=1 / self.scale_factor, mode="bilinear")

        # Estimation
        f1 = self.act(self.conv1(x_down))
        f2 = self.act(self.conv2(f1))
        f3 = self.act(self.conv3(f2))
        f4 = self.act(self.conv4(f3))
        f5 = self.act(self.conv5(torch.cat([f3, f4], dim=1)))
        f6 = self.act(self.conv6(torch.cat([f2, f5], dim=1)))
        a  =   F.tanh(self.conv7(torch.cat([f1, f6], dim=1)))
        
        # Upsampling
        if self.scale_factor != 1:
            a = self.upsample(a)

        # Enhancement
        if self.out_channels == 3:
            y = x
            for _ in range(self.num_iters):
                y = y + a * (torch.pow(y, 2) - y)
        else:
            y = x
            A = torch.split(a, 3, dim=1)
            for i in range(self.num_iters):
                y = y + A[i] * (torch.pow(y, 2) - y)

        # Unsharp masking
        if self.unsharp_sigma is not None:
            y = kornia.filters.unsharp_mask(y, (3, 3), (self.unsharp_sigma, self.unsharp_sigma))

        return a, y

    def forward_once_variant(
        self,
        input    : torch.Tensor,
        profile  : bool = False,
        out_index: int  = -1,
        *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass once. Implement the logic for a single forward pass. Mainly used for ablation study.

        Args:
            input: An input of shape :math:`[N, C, H, W]`.
            profile: Measure processing time. Default: ``False``.
            out_index: Return specific layer's output from :param:`out_index`.
                Default: ``-1`` means the last layer.

        Return:
            Predictions.
        """
        x = input

        # Downsampling
        x_down = x
        if self.scale_factor != 1:
            x_down = F.interpolate(x, scale_factor=1 / self.scale_factor, mode="bilinear")

        # Estimation
        # if self.variant[0:2] in ["10"]:
        f1  = self.act(self.conv1(x_down))
        f2  = self.act(self.conv2(f1))
        f3  = self.act(self.conv3(f2))
        f4  = self.act(self.conv4(f3))
        f5  = self.act(self.conv5(torch.cat([f3, f4], dim=1)))
        f6  = self.act(self.conv6(torch.cat([f2, f5], dim=1)))
        a   =   F.tanh(self.conv7(torch.cat([f1, f6], dim=1)))

        # Upsampling
        if self.scale_factor != 1:
            a = self.upsample(a)

        # Enhancement
        # Default
        if self.variant[3] == "0":
            if self.out_channels == 3:
                y = x
                for _ in range(self.num_iters):
                    y = y + a * (torch.pow(y, 2) - y)
            else:
                y = x
                A = torch.split(a, 3, dim=1)
                for i in range(self.num_iters):
                    y = y + A[i] * (torch.pow(y, 2) - y)
        # Global P
        elif self.variant[3] == "1":
            if self.out_channels == 3:
                y = x
                p = prior.get_guided_brightness_enhancement_map_prior(x, self.gamma, 9)
                for _ in range(self.num_iters):
                    b = y * (1 - p)
                    d = y * p
                    y = b + d + a * (torch.pow(d, 2) - d)
            else:
                y = x
                A = torch.split(a, 3, dim=1)
                p = prior.get_guided_brightness_enhancement_map_prior(x, self.gamma, 9)
                for i in range(self.num_iters):
                    b = y * (1 - p)
                    d = y * p
                    y = b + d + A[i] * (torch.pow(d, 2) - d)
        # Global P Inference Only
        elif self.variant[3] == "2":
            if self.out_channels == 3:
                if self.phase == ModelPhase.TRAINING:
                    y = x
                    for _ in range(self.num_iters):
                        y = y + a * (torch.pow(y, 2) - y)
                else:
                    y = x
                    p = prior.get_guided_brightness_enhancement_map_prior(x, self.gamma, 9)
                    for _ in range(self.num_iters):
                        b = y * (1 - p)
                        d = y * p
                        y = b + d + a * (torch.pow(d, 2) - d)
            else:
                if self.phase == ModelPhase.TRAINING:
                    y = x
                    A = torch.split(a, 3, dim=1)
                    for i in range(self.num_iters):
                        y = y + A[i] * (torch.pow(y, 2) - y)
                else:
                    y = x
                    A = torch.split(a, 3, dim=1)
                    p = prior.get_guided_brightness_enhancement_map_prior(x, self.gamma, 9)
                    for i in range(self.num_iters):
                        b = y * (1 - p)
                        d = y * p
                        y = b + d + A[i] * (torch.pow(d, 2) - d)
        # Iterative P Inference Only
        elif self.variant[3] == "3":
            if self.out_channels == 3:
                if self.phase == ModelPhase.TRAINING:
                    y = x
                    for _ in range(self.num_iters):
                        y = y + a * (torch.pow(y, 2) - y)
                else:
                    y = x
                    for _ in range(self.num_iters):
                        p = prior.get_guided_brightness_enhancement_map_prior(y, self.gamma, 9)
                        b = y * (1 - p)
                        d = y * p
                        y = b + d + a * (torch.pow(d, 2) - d)
            else:
                if self.phase == ModelPhase.TRAINING:
                    y = x
                    A = torch.split(a, 3, dim=1)
                    for i in range(self.num_iters):
                        y = y + A[i] * (torch.pow(y, 2) - y)
                else:
                    y = x
                    A = torch.split(a, 3, dim=1)
                    for i in range(self.num_iters):
                        p = prior.get_guided_brightness_enhancement_map_prior(y, self.gamma, 9)
                        b = y * (1 - p)
                        d = y * p
                        y = b + d + A[i] * (torch.pow(d, 2) - d)

        # Unsharp masking
        if self.unsharp_sigma is not None:
            y = kornia.filters.unsharp_mask(y, (3, 3), (self.unsharp_sigma, self.unsharp_sigma))

        #
        return a, y

    def regularization_loss(self, alpha: float = 0.1):
        loss = 0.0
        for sub_module in [
            self.conv1, self.conv2, self.conv3, self.conv4, self.conv5,
            self.conv6, self.conv7
        ]:
            if hasattr(sub_module, "regularization_loss"):
                loss += sub_module.regularization_loss()
        return alpha * loss

# endregion
