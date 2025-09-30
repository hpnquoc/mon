#!/usr/bin/env python
# -*- coding: utf-8 -*-

__all__ = [
    "Loss",
]

from typing import Literal

import torch

from mon.core import log, nn
from mon.core.nn import functional as F


class Loss(nn.BaseLoss):
    
    def __init__(
        self,
        loss_e_mean  : float = 0.1,
        loss_w_f     : float = 1.0,
        loss_w_s     : float = 5.0,
        loss_w_e     : float = 8.0,
        loss_w_tv    : float = 20.0,
        loss_w_de    : float = 10.0,
        required_grad: bool  = True,
        reduction    : Literal["none", "mean", "sum"] = "mean",
        verbose      : bool  = False,
    ):
        super().__init__(reduction=reduction)
        self.loss_w_f   = loss_w_f
        self.loss_w_s   = loss_w_s
        self.loss_w_e   = loss_w_e
        self.loss_w_tv  = loss_w_tv
        self.loss_w_de  = loss_w_de
        self.verbose    = verbose

        self.loss_e     = nn.ExposureValueControlLoss(
            patch_size    = 16,
            mean_val      = loss_e_mean,
            required_grad = required_grad,
            reduction     = reduction
        )
        self.loss_tv    = nn.TotalVariationLoss(reduction=reduction)
        self.loss_depth = nn.DepthAwareIlluminationLoss(reduction=reduction)

    def forward(
        self,
        illu_lr         : torch.Tensor,
        image_i_lr      : torch.Tensor,
        image_i_fixed_lr: torch.Tensor,
        depth_lr        : torch.Tensor = None,
    ) -> torch.Tensor:
        loss_f  = self.loss_w_f  * torch.mean(torch.abs(torch.pow(illu_lr - image_i_lr, 2)))
        loss_s  = self.loss_w_s  * torch.mean(image_i_fixed_lr)
        loss_e  = self.loss_w_e  * torch.mean(self.loss_e(illu_lr))
        loss_tv = self.loss_w_tv * self.loss_tv(illu_lr)
        loss_de = 0.0
        if depth_lr is not None:
            loss_de = self.loss_depth(illu_lr, depth_lr)
        loss_de = self.loss_w_de * loss_de
        loss    = loss_f + loss_s + loss_e + loss_tv + loss_de
        
        if self.verbose:
            log(
                f"loss_f : {loss_f:.4f}, "
                f"loss_s : {loss_s:.4f}, "
                f"loss_e : {loss_e:.4f}, "
                f"loss_tv: {loss_tv:.4f}, "
                f"loss_de: {loss_de:.4f}, "
            )
        
        return loss


class L_edge(nn.BaseLoss):
    """Computes edge-aware loss to penalize blurring by ensuring enhanced
    gradients are at least as strong as input.
    """
    
    def __init__(self, reduction: str = "mean"):
        super().__init__(reduction=reduction)
    
    def sobel_gradient(self, image: torch.Tensor) -> torch.Tensor:
        """Computes Sobel gradient magnitude for the input image."""
        # Assume image is (b, c, h, w) with c=1 for Value component
        sobel_x = torch.tensor(
            data=[[-1, 0, 1],
                  [-2, 0, 2],
                  [-1, 0, 1]],
            dtype=image.dtype, device=image.device).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor(
            data=[[-1, -2, -1],
                  [ 0,  0,  0],
                  [ 1,  2,  1]],
            dtype=image.dtype, device=image.device).unsqueeze(0).unsqueeze(0)
        grad_x = F.conv2d(image, sobel_x, padding=1)
        grad_y = F.conv2d(image, sobel_y, padding=1)
        grad   = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)
        return grad
    
    def forward(self, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        sobel_y = self.sobel_gradient(y)
        sobel_z = self.sobel_gradient(z)
        # Penalize where enhanced has weaker edges
        loss    = torch.max(torch.zeros_like(sobel_y), sobel_y - sobel_z)
        loss    = self.reduce(loss)
        return loss
