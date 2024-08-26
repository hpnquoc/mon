#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Zero-DiDCE.

This module implements the paper: "Zero-Reference Dual-Illumination Deep Curve
Estimation".

References:
    https://github.com/Wenhui-Luo/Zero-DiDCE
"""

from __future__ import annotations

__all__ = [
    "ZeroDiDCE_RE",
]

from typing import Any, Literal

import torch

from mon import core, nn
from mon.globals import MODELS, Scheme, Task
from mon.vision.enhance import base

console      = core.console
current_file = core.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# region Loss

class Loss(nn.Loss):

    def __init__(
        self,
        spa_weight    : float = 1.0,
        exp_patch_size: int   = 16,
        exp_mean_val  : float = 0.6,
        exp_weight    : float = 10.0,
        col_weight    : float = 5.0,
        tva_weight    : float = 200.0,
        reduction     : Literal["none", "mean", "sum"] = "mean",
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.spa_weight = spa_weight
        self.exp_weight = exp_weight
        self.col_weight = col_weight
        self.tva_weight = tva_weight
        
        self.loss_spa = nn.SpatialConsistencyLoss(reduction=reduction)
        self.loss_exp = nn.ExposureControlLoss(
            reduction  = reduction,
            patch_size = exp_patch_size,
            mean_val   = exp_mean_val,
        )
        self.loss_col = nn.ColorConstancyLoss(reduction=reduction)
        self.loss_tva = nn.TotalVariationLoss(reduction=reduction)
    
    def forward(
        self,
        input  : torch.Tensor,
        adjust : torch.Tensor,
        enhance: torch.Tensor,
        **_
    ) -> torch.Tensor:
        loss_spa = self.loss_spa(input=enhance, target=input)
        loss_exp = self.loss_exp(input=enhance)
        loss_col = self.loss_col(input=enhance)
        loss_tva = self.loss_tva(input=adjust)
        loss     = (
              self.spa_weight * loss_spa
            + self.exp_weight * loss_exp
            + self.col_weight * loss_col
            + self.tva_weight * loss_tva
        )
        return loss

# endregion


# region Model

@MODELS.register(name="zero_didce_re", arch="zero_didce")
class ZeroDiDCE_RE(base.ImageEnhancementModel):
    """Zero-Reference Dual-Illumination Deep Curve Estimation model.
    
    Args:
        in_channels: The first layer's input channel. Default: ``3`` for RGB
            image.
        num_channels: The number of input and output channels for subsequent
            layers. Default: ``32``.
        num_iters: The number of progressive loop. Default: ``8``.
        
    References:
        https://github.com/Wenhui-Luo/Zero-DiDCE
    """
    
    model_dir: core.Path    = current_dir
    arch     : str          = "zero_didce"
    tasks    : list[Task]   = [Task.LLIE]
    schemes  : list[Scheme] = [Scheme.ZERO_REFERENCE]
    zoo      : dict         = {}

    def __init__(
        self,
        in_channels : int = 3,
        num_channels: int = 32,
        weights     : Any = None,
        *args, **kwargs
    ):
        super().__init__(
            name        = "zero_didce",
            in_channels = in_channels,
            weights     = weights,
            *args, **kwargs
        )
        
        # Populate hyperparameter values from pretrained weights
        if isinstance(self.weights, dict):
            in_channels  = self.weights.get("in_channels" , in_channels)
            num_channels = self.weights.get("num_channels", num_channels)
        self.in_channels  = in_channels
        self.num_channels = num_channels
        
        # Construct model
        self.relu     = nn.ReLU(inplace=True)
        self.e_conv1  = nn.Conv2d(self.in_channels,      self.num_channels, 3, 1, 1, bias=True)
        self.e_conv2  = nn.Conv2d(self.num_channels,     self.num_channels, 3, 1, 1, bias=True)
        self.e_conv3  = nn.Conv2d(self.num_channels,     self.num_channels, 3, 1, 1, bias=True)
        self.e_conv7  = nn.Conv2d(self.num_channels * 2, self.out_channels, 3, 1, 1, bias=True)
        self.maxpool  = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        
        # Loss
        self.loss = Loss()
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)

    def init_weights(self, m: nn.Module):
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            if hasattr(m, "conv"):
                m.conv.weight.data.normal_(0.0, 0.02)
            else:
                m.weight.data.normal_(0.0, 0.02)
        elif classname.find("BatchNorm") != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
    
    def forward_loss(self, datapoint: dict, *args, **kwargs) -> dict | None:
        # Forward
        outputs  = self.forward(datapoint=datapoint, *args, **kwargs)
        self.assert_datapoint(datapoint)
        self.assert_outputs(outputs)
        # Loss
        image    = datapoint.get("image")
        enhanced = outputs.get("enhanced")
        adjust   = outputs.get("adjust")
        outputs["loss"] = self.loss(image, adjust, enhanced)
        # Return
        return outputs
    
    def forward(self, datapoint: dict, *args, **kwargs) -> dict:
        self.assert_datapoint(datapoint)
        x    = datapoint.get("image")
        #
        x1   =  self.relu(self.e_conv1(x))
        x2   =  self.relu(self.e_conv2(x1))
        x3   =  self.relu(self.e_conv3(x2))
        x_r  = torch.tanh(self.e_conv7(torch.cat([x1, x3], 1)))
        #
        xx   = 1 - x
        x11  =  self.relu(self.e_conv1(xx))
        x21  =  self.relu(self.e_conv2(x11))
        x31  =  self.relu(self.e_conv3(x21))
        x_r1 = torch.tanh(self.e_conv7(torch.cat([x11, x31], 1)))
        #
        x_r  = (x_r + x_r1) / 2
        #
        xx1  = torch.mean(x).item()
        n1   = 0.63
        s    = xx1 * xx1
        n3   = -0.79 * s + 0.81 * xx1 + 1.4
        if xx1 < 0.1:
            b = -25 * xx1 + 10
        elif xx1 < 0.45:
            b = 17.14 * s - 15.14 * xx1 + 10
        else:
            b = 5.66 * s - 2.93 * xx1 + 7.2
        #
        b = int(b)
        y = x
        for i in range(b):
            y = y + x_r * (torch.pow(y, 2) - y) * ((n1 - torch.mean(y).item()) / (n3 - torch.mean(y).item()))
        #
        return {
            "adjust"  : x_r,
            "enhanced": y,
        }
    
# endregion
