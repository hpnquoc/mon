#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""ZS-N2N.

This module implements the paper: "Zero-Shot Noise2Noise: Efficient Image
Denoising without any Data".

References:
    - https://colab.research.google.com/drive/1i82nyizTdszyHkaHBuKPbWnTzao8HF9b?usp=sharing#scrollTo=Srf0GQTYrkxA
"""

from __future__ import annotations

__all__ = [
    "ZSN2N",
]

import torch
from torch.nn.common_types import _size_2_t

from mon import core, nn
from mon.globals import MODELS, LType, Task
from mon.vision import dtype, geometry
from mon.vision.enhance import base

console      = core.console
current_file = core.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# region Model

@MODELS.register(name="zsn2n", arch="zsn2n")
class ZSN2N(base.ImageEnhancementModel):
    """Zero-Shot Noise2Noise: Efficient Image Denoising without any Data.
    
    Args:
        in_channels: The first layer's input channel. Default: ``3`` for RGB
            image.
        num_channels: Output channels for subsequent layers. Default: ``48``.
    
    References:
        - https://colab.research.google.com/drive/1i82nyizTdszyHkaHBuKPbWnTzao8HF9b?usp=sharing#scrollTo=Srf0GQTYrkxA
    """
    
    arch     : str          = "zsn2n"
    name     : str          = "zsn2n"
    tasks    : list[Task]   = [Task.DENOISE]
    ltypes   : list[LType]  = [LType.ZERO_SHOT]
    model_dir: core.Path    = current_dir
    zoo      : dict         = {}
    
    def __init__(
        self,
        in_channels : int = 3,
        num_channels: int = 48,
        iters       : int = 3000,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.iters = iters
        
        # Network
        self.conv1 = nn.Conv2d(in_channels,  num_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(num_channels, in_channels,  kernel_size=1)
        self.act   = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
        # Optimizer
        self.configure_optimizers()
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)
        self.initial_state_dict = self.state_dict()
    
    def init_weights(self, m: nn.Module):
        pass
    
    def configure_optimizers(self):
        return None
        
    def forward_loss(self, datapoint: dict, *args, **kwargs) -> dict:
        # Forward
        noisy                = datapoint["image"]
        noisy1, noisy2       = self.pair_downsampler(noisy)
        datapoint1           = datapoint | {"image": noisy1}
        datapoint2           = datapoint | {"image": noisy2}
        outputs1             = self.forward(datapoint=datapoint1, *args, **kwargs)
        outputs2             = self.forward(datapoint=datapoint2, *args, **kwargs)
        outputs              = self.forward(datapoint=datapoint,  *args, **kwargs)
        # Symmetric Loss
        pred1                = noisy1 - outputs1["enhanced"]
        pred2                = noisy2 - outputs2["enhanced"]
        noisy_denoised       =  noisy -  outputs["enhanced"]
        denoised1, denoised2 = self.pair_downsampler(noisy_denoised)
        mse_loss  = nn.MSELoss()
        loss_res  = 0.5 * (mse_loss(noisy1, pred2)    + mse_loss(noisy2, pred1))
        loss_cons = 0.5 * (mse_loss(pred1, denoised1) + mse_loss(pred2, denoised2))
        loss      = loss_res + loss_cons
        # loss      = nn.reduce_loss(loss=loss, reduction="mean")
        outputs["loss"] = loss
        # Return
        return outputs
    
    def forward(self, datapoint: dict, *args, **kwargs) -> dict:
        x = datapoint["image"]
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        y = self.conv3(x)
        if self.predicting:
            y = torch.clamp(y, 0, 1)
        return {"enhanced": y}
  
    def infer(
        self,
        datapoint    : dict,
        image_size   : _size_2_t = 512,
        resize       : bool      = False,
        reset_weights: bool      = True,
    ) -> dict:
        """Infer the model on a single datapoint. This method is different from
        `forward()` in term that you may want to perform additional
        pre-processing or post-processing steps.
        
        Notes:
            If you want to perform specific pre-processing or post-processing
            steps, you should override this method.
        
        Args:
            datapoint: A `dict` containing the attributes of a datapoint.
            image_size: The input size. Default: ``512``.
            resize: Resize the input image to the model's input size. Default: ``False``.
            reset_weights: Whether to reset the weights before training. Default: ``True``.
        """
        # Initialize training components
        if reset_weights:
            self.load_state_dict(self.initial_state_dict, strict=False)
        optimizer    = self.optimizer.get("optimizer",    None)
        lr_scheduler = self.optimizer.get("lr_scheduler", {})
        scheduler    =   lr_scheduler.get("scheduler",    None)
        optimizer = optimizer or nn.Adam(self, lr=1e-3, weight_decay=0.0001)
        scheduler = scheduler or nn.StepLR(optimizer, step_size=1000, gamma=0.5)
        
        # Input
        image  = datapoint["image"].to(self.device)
        h0, w0 = dtype.get_image_size(image)
        if resize:
            image = geometry.resize(image, image_size)
        else:
            image = geometry.resize(image, divisible_by=32)
        
        # Optimize
        timer = core.Timer()
        timer.tick()
        self.train()
        for _ in range(self.iters):
            outputs = self.forward_loss(datapoint={"image": image})
            optimizer.zero_grad()
            loss = outputs["loss"]
            loss.backward(retain_graph=True)
            optimizer.step()
            scheduler.step()
        self.eval()
        outputs = self.forward(datapoint={"image": image})
        timer.tock()
        
        # Post-processing
        enhanced = outputs["enhanced"]
        enhanced = geometry.resize(enhanced, (h0, w0))
        
        # Return
        return outputs | {
            "enhanced": enhanced,
            "time"    : timer.avg_time,
        }
        
# endregion
