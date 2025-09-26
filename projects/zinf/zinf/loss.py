#!/usr/bin/env python
# -*- coding: utf-8 -*-

__all__ = [
    "L_exp",
    "L_tv",
    "Loss",
]

from typing import Literal

import torch

from mon.core import log, nn


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


class L_exp(nn.Module):

    def __init__(self, patch_size: int, mean_val: float):
        super().__init__()
        self.pool     = nn.AvgPool2d(patch_size)
        self.mean_val = mean_val

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = self.pool(x) ** 0.5
        d    = torch.abs(torch.mean(torch.pow(mean - torch.FloatTensor([self.mean_val]).to(x.device), 2)))
        return d


class L_tv(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        b, c, h, w = x.shape
        count_h    = (x.size()[2] - 1) * x.size()[3]
        count_w    = x.size()[2] * (x.size()[3] - 1)
        h_tv       = torch.pow((x[:, :, 1:, :] - x[:, :, :h - 1, :]), 2).sum()
        w_tv       = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w - 1]), 2).sum()
        return 2 * (h_tv / count_h + w_tv / count_w) / b
