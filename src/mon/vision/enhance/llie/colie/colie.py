#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""CoLIE.

This module implements the paper: "Fast Context-Based Low-Light Image
Enhancement via Neural Implicit Representations," ECCV 2024.

References:
    https://github.com/ctom2/colie
"""

from __future__ import annotations

__all__ = [
    "CoLIE_RE",
]

from typing import Any, Literal

import cv2
import numpy as np
import torch
from mon import core, nn
from mon.globals import MODELS, Scheme, Task
from mon.vision import filtering
from mon.vision.enhance import base
from torch.nn import functional as F

console      = core.console
current_file = core.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# region Loss

class Loss(nn.Loss):
    
    def __init__(
        self,
        L     : float = 0.5,
        alpha : float = 1,
        beta  : float = 20,
        gamma : float = 8,
        delta : float = 5,
        loss_weight: float = 1.0,
        reduction  : Literal["none", "mean", "sum"] = "mean",
        *args, **kwargs
    ):
        super().__init__(loss_weight=loss_weight, reduction=reduction)
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma
        self.delta = delta
        self.l_exp = nn.ExposureValueControlLoss(patch_size=16, mean_val=L)
        self.l_tv  = nn.TotalVariationLoss()
        
    def forward(
        self,
        illu_lr         : torch.Tensor,
        image_v_lr      : torch.Tensor,
        image_v_fixed_lr: torch.Tensor,
    ) -> torch.Tensor:
        loss_spa      = torch.mean(torch.abs(torch.pow(illu_lr - image_v_lr, 2)))
        loss_tv       = self.l_tv(illu_lr)
        loss_exp      = torch.mean(self.l_exp(illu_lr))
        loss_sparsity = torch.mean(image_v_fixed_lr)
        loss = (
                   loss_spa * self.alpha
                  + loss_tv * self.beta
                 + loss_exp * self.gamma
            + loss_sparsity * self.delta
        )
        return loss

# endregion


# region Module

class SirenLayer(nn.Module):
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        w0          : int  = 30,
        is_first    : bool = False,
        is_last     : bool = False
    ):
        super().__init__()
        self.in_channels = in_channels
        self.w0          = w0
        self.linear      = nn.Linear(in_channels, out_channels)
        self.is_first    = is_first
        self.is_last     = is_last
        if not self.is_last:
            self.init_weights()
    
    def init_weights(self):
        if self.is_first:
            b = 1 / self.in_channels
        else:
            b = np.sqrt(6 / self.in_channels) / self.w0
        with torch.no_grad():
            self.linear.weight.uniform_(-b, b)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        x = self.linear(x)
        y = nn.Sigmoid()(x) if self.is_last else torch.sin(self.w0 * x)
        return y

# endregion


# region Model

@MODELS.register(name="colie_re", arch="colie")
class CoLIE_RE(base.ImageEnhancementModel):
    """"Fast Context-Based Low-Light Image Enhancement via Neural Implicit
    Representations," ECCV 2024.
    
    Args:
        window_size: Context window size. Default: ``1``.
        down_size  : Downsampling size. Default: ``256``.
        add_layer: Should be in range of  ``[1, :obj:`num_layers` - 2]``.
        L: The "optimally-intense threshold", lower values produce brighter
            images. Default: ``0.3``.
        alpha: Fidelity control. Default: ``1``.
        beta: Illumination smoothness. Default: ``20``.
        gamma: Exposure control. Default: ``8``.
        delta: Sparsity level. Default: ``5``.
    
    References:
        https://github.com/ctom2/colie
    """
    
    model_dir: core.Path    = current_dir
    arch     : str          = "colie"
    tasks    : list[Task]   = [Task.LLIE]
    schemes  : list[Scheme] = [Scheme.ZERO_REFERENCE, Scheme.INSTANCE]
    zoo      : dict         = {}
    
    def __init__(
        self,
        window_size : int   = 1,
        down_size   : int   = 256,
        num_layers  : int   = 4,
        hidden_dim  : int   = 256,
        add_layer   : int   = 2,
        weight_decay: list[float] = [0.1, 0.0001, 0.001],
        L           : float = 0.3,
        alpha       : float = 1,
        beta        : float = 20,
        gamma       : float = 8,
        delta       : float = 5,
        weights     : Any   = None,
        *args, **kwargs
    ):
        super().__init__(
            name    = "colie_re",
            weights = weights,
            *args, **kwargs
        )
        self.window_size = window_size
        self.patch_dim   = window_size ** 2
        self.down_size   = down_size
        self.omega_0     = 30.0
        self.siren_C     = 6.0
        
        patch_layers   = [nn.SIREN(self.patch_dim, hidden_dim, self.omega_0, self.siren_C, is_first=True)]
        spatial_layers = [nn.SIREN(2,   hidden_dim, self.omega_0, self.siren_C, is_first=True)]
        for _ in range(1, add_layer - 2):
            patch_layers.append(  nn.SIREN(hidden_dim, hidden_dim, self.omega_0, self.siren_C))
            spatial_layers.append(nn.SIREN(hidden_dim, hidden_dim, self.omega_0, self.siren_C))
        patch_layers.append(  nn.SIREN(hidden_dim, hidden_dim // 2, self.omega_0, self.siren_C))
        spatial_layers.append(nn.SIREN(hidden_dim, hidden_dim // 2, self.omega_0, self.siren_C))
        
        output_layers  = []
        for _ in range(add_layer, num_layers - 1):
            output_layers.append(nn.SIREN(hidden_dim, hidden_dim, self.omega_0, self.siren_C))
        output_layers.append(nn.SIREN(hidden_dim, 1, self.omega_0, self.siren_C, is_last=True))

        self.patch_net   = nn.Sequential(*patch_layers)
        self.spatial_net = nn.Sequential(*spatial_layers)
        self.output_net  = nn.Sequential(*output_layers)
        
        if not weight_decay:
            weight_decay = [0.1, 0.0001, 0.001]
            
        self.params = []
        self.params += [{
	        "params"      : self.spatial_net.parameters(),
	        "weight_decay": weight_decay[0]
        }]
        self.params += [{
	        "params"      : self.patch_net.parameters(),
	        "weight_decay": weight_decay[1]
        }]
        self.params += [{
	        "params"      : self.output_net.parameters(),
	        "weight_decay": weight_decay[2]
        }]
        
        # Loss
        self.loss = Loss(L, alpha, beta, gamma, delta)
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)
        self.initial_state_dict = self.state_dict()
    
    def init_weights(self, m: nn.Module):
        pass
    
    def forward_loss(self, datapoint: dict, *args, **kwargs) -> dict:
        # Forward
        self.assert_datapoint(datapoint)
        outputs = self.forward(datapoint=datapoint, *args, **kwargs)
        self.assert_outputs(outputs)
        # Loss
        illu_lr          = outputs["illu_lr"]
        image_v_lr       = outputs["image_v_lr"]
        image_v_fixed_lr = outputs["image_v_fixed_lr"]
        outputs["loss"]  = self.loss(illu_lr, image_v_lr, image_v_fixed_lr)
        # Return
        return outputs
        
    def forward(self, datapoint: dict, *args, **kwargs) -> dict:
        self.assert_datapoint(datapoint)
        image_rgb   = datapoint.get("image")
        image_hsv   = core.rgb_to_hsv(image_rgb)
        image_v     = core.rgb_to_v(image_rgb)
        image_v_lr  = self.interpolate_image(image_v)
        patch       = self.get_patches(image_v_lr)
        spatial     = self.get_coords()
        
        illu_res_lr = self.output_net(
            torch.cat(
                (self.patch_net(patch), self.spatial_net(spatial)), -1
            )
        )
        illu_res_lr = illu_res_lr.view(1, 1, self.down_size, self.down_size)
        illu_lr     = illu_res_lr + image_v_lr
        
        image_v_fixed_lr = image_v_lr / (illu_lr + 1e-4)
        image_v_fixed    = self.filter_up(image_v_lr, image_v_fixed_lr, image_v)
        image_hsv_fixed  = self.replace_v_component(image_hsv, image_v_fixed)
        image_rgb_fixed  = core.hsv_to_rgb(image_hsv_fixed)
        image_rgb_fixed  = image_rgb_fixed / torch.max(image_rgb_fixed)
        
        return {
            # "image_hsv"       : image_hsv,
            "image_v"         : image_v,
            "image_v_lr"      : image_v_lr,
            # "patch"           : patch,
            # "spatial"         : spatial,
            # "illu_res_lr"     : illu_res_lr,
            "illu_lr"         : illu_lr,
            "image_v_fixed_lr": image_v_fixed_lr,
            "image_v_fixed"   : image_v_fixed,
            # "image_hsv_fixed" : image_hsv_fixed,
            "enhanced"        : image_rgb_fixed,
        }
    
    def interpolate_image(self, image: torch.Tensor) -> torch.Tensor:
        """Reshapes the image based on new resolution."""
        return F.interpolate(image, size=(self.down_size, self.down_size))
    
    def get_patches(self, image: torch.Tensor) -> torch.Tensor:
        """Creates a tensor where the channel contains patch information."""
        kernel = torch.zeros(
            (self.window_size ** 2, 1, self.window_size, self.window_size)
        ).cuda()
        
        for i in range(self.window_size):
            for j in range(self.window_size):
                kernel[int(torch.sum(kernel).item()), 0, i, j] = 1
    
        pad       = nn.ReflectionPad2d(self.window_size // 2)
        im_padded = pad(image)
        extracted = F.conv2d(im_padded, kernel, padding=0).squeeze(0)
        return torch.movedim(extracted, 0, -1)
    
    def get_coords(self) -> torch.Tensor:
        """Creates a coordinates grid for INF."""
        coords = np.dstack(
            np.meshgrid(
                np.linspace(0, 1, self.down_size),
                np.linspace(0, 1, self.down_size)
            )
        )
        coords = torch.from_numpy(coords).float().cuda()
        return coords
    
    @staticmethod
    def filter_up(
        x_lr  : torch.Tensor,
        y_lr  : torch.Tensor,
        x_hr  : torch.Tensor,
        radius: int = 1
    ):
        """Applies the guided filter to upscale the predicted image. """
        gf   = filtering.FastGuidedFilter(radius=radius)
        y_hr = gf(x_lr, y_lr, x_hr)
        y_hr = torch.clip(y_hr, 0, 1)
        return y_hr
    
    @staticmethod
    def replace_v_component(
        image_hsv: torch.Tensor, v_new: torch.Tensor
    ) -> torch.Tensor:
        """Replaces the `V` component of an HSV image `[1, 3, H, W]`."""
        image_hsv[:, -1, :, :] = v_new
        return image_hsv
    
    def infer(
        self,
        datapoint    : dict,
        epochs       : int   = 200,
        lr           : float = 1e-5,
        weight_decay : float = 3e-4,
        reset_weights: bool  = True,
        *args, **kwargs
    ) -> dict:
        # Initialize training components
        self.train()
        if reset_weights:
            self.load_state_dict(self.initial_state_dict)
        if isinstance(self.optims, dict):
            optimizer = self.optims.get("optimizer", None)
        else:
            optimizer = nn.Adam(
                self.parameters(),
                lr           = lr,
                betas        = (0.9, 0.999),
                weight_decay = weight_decay,
            )
        
        # Pre-processing
        self.assert_datapoint(datapoint)
        for k, v in datapoint.items():
            if isinstance(v, torch.Tensor):
                datapoint[k] = v.to(self.device)
        
        # Training
        for _ in range(epochs):
            outputs = self.forward_loss(datapoint=datapoint)
            optimizer.zero_grad()
            loss = outputs["loss"]
            loss.backward(retain_graph=True)
            optimizer.step()
            # if self.verbose:
            #    console.log(f"Loss: {loss.item()}")
            
        # Forward
        self.eval()
        timer = core.Timer()
        timer.tick()
        outputs = self.forward(datapoint=datapoint)
        timer.tock()
        self.assert_outputs(outputs)
    
        # Return
        outputs["time"] = timer.avg_time
        return outputs
    
# endregion


# region Main

def run_colie():
    path      = core.Path("./data/00691.png")
    image     = cv2.imread(str(path))
    datapoint = {"image": core.to_image_tensor(image, False, True)}
    device    = torch.device("cuda:0")
    net       = CoLIE_RE().to(device)
    outputs   = net.infer(datapoint)
    enhanced  = outputs.get("enhanced")
    enhanced  = core.to_image_nparray(enhanced, False, True)
    cv2.imshow("Image",    image)
    cv2.imshow("Enhanced", enhanced)
    cv2.waitKey(0)


if __name__ == "__main__":
    run_colie()

# endregion
