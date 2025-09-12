#!/usr/bin/env python
# -*- coding: utf-8 -*-

__all__ = [
    "ISPLoss",
    "LLELoss",
    "WarmupLoss",
]

import torch

from mon.core import nn


class OutlierAwareLoss(nn.BaseLoss):
    
    def __init__(self, reduction: str = "mean"):
        super().__init__(reduction=reduction)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        delta  = input - target
        var    = delta.std((2, 3), keepdims=True) / (2 ** 0.5)
        avg    = delta.mean((2, 3), True)
        weight = torch.tanh((delta - avg).abs() / (var + 1e-6)).detach()       
        loss   = (delta.abs() * weight)
        loss   = self.reduce(loss=loss)
        return loss
    

class WarmupLoss(nn.BaseLoss):
    
    def __init__(self, reduction: str = "mean"):
        super().__init__(reduction=reduction)
        self.loss_cb = nn.CharbonnierLoss(1e-8, reduction=reduction)
        self.loss_cs = nn.CosineSimilarity(reduction=reduction)

    def forward(self, input, target, warmup1, warmup2):
        loss = (self.loss_cb(warmup2, input) +
                (self.loss_cb(warmup1, target)
                 + (1 - self.loss_cs(warmup1.clip(0, 1), target))))
        loss = self.reduce(loss=loss)
        return loss 


class LLELoss(nn.BaseLoss):
    
    def __init__(self, reduction: str = "mean"):
        super().__init__(reduction=reduction)
        self.loss_cs = nn.CosineSimilarity(reduction=reduction)
        self.loss_oa = OutlierAwareLoss(reduction=reduction)
        self.psnr    = nn.PSNRLoss(reduction=reduction)
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = ((self.loss_oa(input, target)
                + (1 - self.loss_cs(input.clip(0, 1), target)))
                + 2 * self.psnr(input, target))
        loss = self.reduce(loss=loss)
        return loss
        
        
class ISPLoss(nn.BaseLoss):
    
    def __init__(self, reduction: str = "mean"):
        super().__init__(reduction=reduction)
        self.loss_cs = nn.CosineSimilarity(reduction=reduction)
        self.loss_oa = OutlierAwareLoss(reduction=reduction)
        self.psnr    = nn.PSNRLoss(reduction=reduction)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = ((self.loss_oa(input, target)
                + (1 - self.loss_cs(input.clip(0, 1), target)))
                + 2 * self.psnr(input, target))
        loss = self.reduce(loss=loss)
        return loss
