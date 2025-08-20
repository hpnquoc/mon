#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements base loss and basic loss functions."""

__all__ = [
    "BaseLoss",
    "CharbonnierLoss",
    "CosineSimilarityLoss",
    "ExtendedL1Loss",
]

import abc
from typing import Literal

import torch
from torch.nn.modules.loss import _Loss, L1Loss

from mon.core.utils import depascalize


# ----- Base Loss -----
class BaseLoss(_Loss, abc.ABC):
    """The base class for all loss functions.
    
    Args:
        reduction: Specifies the reduction to apply to the output. One of:
            - ``'none'``: No reduction will be applied.
            - ``'mean'``: The sum of the output will be divided by the number of
                elements in the output.
            - ``'sum'``: The output will be summed.
            - Default: ``'mean'``.
    """
    
    reductions = ["none", "mean", "sum"]
    
    def __init__(self, reduction: Literal["none", "mean", "sum"] = "mean"):
        super().__init__(reduction=reduction)
        if self.reduction not in self.reductions:
            raise ValueError(f"[reduction] must be one of: {self.reductions}, got {reduction}.")
        
    def __str__(self):
        return depascalize(self.__class__.__name__).lower()
    
    @abc.abstractmethod
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate the loss between input and target.
    
        Args:
            input: Input data as a ``torch.Tensor`` of shape :math:`(B, C, H, W)` in range :math:`[0.0, 1.0]`.
            target: Target data as a ``torch.Tensor`` of shape :math:`(B, C, H, W)` in range :math:`[0.0, 1.0]`.
    
        Returns:
            Loss value as a ``torch.Tensor``.
        """
        pass
    
    def reduce(self, loss: torch.Tensor) -> torch.Tensor:
        """Reduces the loss tensor.
    
        Args:
            loss: Elementwise loss tensor as a ``torch.Tensor``.
    
        Returns:
            Reduced loss valued as a ``torch.Tensor``.
        """
        return {"mean": torch.mean, "sum": torch.sum, "none": lambda x: x}[self.reduction](loss)


# ----- Basic Loss -----
class CharbonnierLoss(BaseLoss):
    """Computes the Charbonnier loss between input and target tensors.

    Args:
        eps: Small constant for numerical stability. Default: ``1e-3``.
        reduction: Reduction method: ``"none"``, ``"mean"``, or ``"sum"``.
            Default: ``"mean"``.`.
    """
    
    def __init__(self, eps: float = 1e-3, reduction: str = "mean"):
        super().__init__(reduction=reduction)
        self.eps = eps
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = torch.sqrt((input - target) ** 2 + (self.eps * self.eps))
        loss = self.reduce(loss=loss)
        return loss
    

class CosineSimilarityLoss(BaseLoss):
    """Computes cosine similarity loss between input and target tensors.

    Args:
        dim: Dimension for cosine similarity. Default: ``1``.
        eps: Small constant for numerical stability. Default: ``1e-6``.
        reduction: Reduction method: ``"none"``, ``"mean"``, or ``"sum"``.
            Default: ``"mean"``.
    """
    
    def __init__(self, dim: int = 1, eps: float = 1e-6, reduction: str = "mean"):
        super().__init__(reduction=reduction)
        self.cos = torch.nn.CosineSimilarity(dim=dim, eps=eps)
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        b, c, h, w = input.shape
        x    = input.permute(0, 2, 3, 1).view(-1, c)
        y    = target.permute(0, 2, 3, 1).view(-1, c)
        loss = 1.0 - self.cos(x, y).sum() / (1.0 * b * h * w)
        loss = self.reduce(loss=loss)
        return loss
        

class ExtendedL1Loss(BaseLoss):
    """Computes extended L1 loss with mask normalization.

    Args:
        reduction: Reduction method: ``"none"``, ``"mean"``, or ``"sum"``.
            Default: ``"mean"``.
    """
    
    def __init__(self, reduction: str = "mean"):
        super().__init__(reduction=reduction)
        self.loss_l1 = L1Loss()
    
    # noinspection PyMethodOverriding
    def forward(self, input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        norm = self.loss_l1(mask, torch.zeros_like(mask))
        loss = self.loss_l1(mask * input, mask * target) / norm
        loss = self.reduce(loss=loss)
        return loss
