#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements loss functions using image perceptual characteristics."""

from __future__ import annotations

__all__ = [
    "EdgeLoss",
    "MS_SSIMLoss",
    "PSNRLoss",
    "PerceptualLoss",
    "SSIMLoss",
]

from typing import Literal

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models, transforms

from mon.globals import LOSSES
from mon.nn.loss import base


# region Edge

@LOSSES.register(name="edge_loss")
class EdgeLoss(base.Loss):
    """Edge Loss computes the difference in edge features between input and target
    using a Laplacian kernel.

    Args:
        loss_weight: Weight of the loss as ``float``. Default is ``1.0``
        reduction: Reduction method as ``Literal["none", "mean", "sum"]``.
            Default is ``"mean"``.

    Attributes:
        kernel: Gaussian kernel for convolution as ``torch.Tensor``.
        loss: Charbonnier loss function as ``base.CharbonnierLoss``.
    """
    
    def __init__(
        self,
        loss_weight: float = 1.0,
        reduction  : Literal["none", "mean", "sum"] = "mean"
    ):
        super().__init__(loss_weight=loss_weight, reduction=reduction)
        k           = torch.Tensor([[0.05, 0.25, 0.4, 0.25, 0.05]])
        self.kernel = torch.matmul(k.t(), k).unsqueeze(0).repeat(3, 1, 1, 1)
        self.loss   = base.CharbonnierLoss()

    def gauss_conv(self, image: torch.Tensor) -> torch.Tensor:
        """Applies Gaussian convolution to the input image.

        Args:
            image: Input tensor as ``torch.Tensor`` with shape [B, C, H, W].

        Returns:
            Convolved tensor as ``torch.Tensor`` with shape [B, C, H, W].
        """
        b, c, w, h  = self.kernel.shape
        self.kernel = self.kernel.to(image.device)
        image       = F.pad(image, (w // 2, h // 2, w // 2, h // 2), mode="replicate")
        # gauss       = F.conv2d(image, self.kernel, groups=b)  # Old code
        gauss       = F.conv2d(image, self.kernel, groups=c)  # Groups=c for channel-wise convolution
        return gauss
    
    def laplacian_kernel(self, image: torch.Tensor) -> torch.Tensor:
        """Computes the Laplacian edge map using a Gaussian pyramid.

        Args:
            image: Input tensor as ``torch.Tensor`` with shape [B, C, H, W].

        Returns:
            Edge map tensor as ``torch.Tensor`` with shape [B, C, H, W].
        """
        filtered   = self.gauss_conv(image)       # filter
        down       = filtered[:, :, ::2, ::2]     # downsample
        new_filter = torch.zeros_like(filtered)
        new_filter[:, :, ::2, ::2] = down * 4     # upsample
        filtered   = self.gauss_conv(new_filter)  # filter
        diff       = image - filtered
        return diff
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Computes the edge loss between input and target tensors.

        Args:
            input: Input tensor as ``torch.Tensor`` with shape [B, C, H, W].
            target: Target tensor as ``torch.Tensor`` with shape [B, C, H, W].

        Returns:
            Loss value as ``torch.Tensor``.
        """
        edge1 = self.laplacian_kernel(input)
        edge2 = self.laplacian_kernel(target)
        diff  = edge1 - edge2
        loss  = torch.mean(torch.sqrt((diff * diff) + (self.eps * self.eps)))
        loss  = base.reduce_loss(loss=loss, reduction=self.reduction)
        return self.loss_weight * loss

# endregion


# region Perceptual

@LOSSES.register(name="perceptual_loss")
class PerceptualLoss(base.Loss):
    """Perceptual Loss computes feature differences between input and target using a pretrained network.

    Args:
        net: Pretrained network as ``nn.Module`` or ``str``; options: ``"alexnet"``,
            ``"vgg11"``, ``"vgg13"``, ``"vgg16"``, ``"vgg19"``. Default is ``"vgg19"``.
        layers: List of layer indices to extract features from as ``list[str]``.
            Default is ``["26"]``.
        preprocess: Applies normalization if ``True``. Default is ``False``.
        loss_weight: Weight of the loss as ``float``. Default is ``1.0``.
        reduction: Reduction method as ``Literal["none", "mean", "sum"]``. Default is ``"mean"``.
    """
    
    def __init__(
        self,
        net        : nn.Module | str = "vgg19",
        layers     : list  = ["26"],
        preprocess : bool  = False,
        loss_weight: float = 1.0,
        reduction  : Literal["none", "mean", "sum"] = "mean"
    ):
        super().__init__(loss_weight=loss_weight, reduction=reduction)
        self.layers     = layers
        self.preprocess = preprocess
        
        if net in ["alexnet"]:
            net = models.alexnet(weights=models.AlexNet_Weights).features
        elif net in ["vgg11"]:
            net = models.vgg11(weights=models.VGG11_Weights).features
        elif net in ["vgg13"]:
            net = models.vgg13(weights=models.VGG13_Weights).features
        elif net in ["vgg16"]:
            net = models.vgg16(weights=models.VGG16_Weights).features
        elif net in ["vgg19"]:
            net = models.vgg19(weights=models.VGG19_Weights).features
        
        self.net     = net.eval()
        self.l1_loss = base.L1Loss(reduction=reduction)
        
        # Disable gradient computation for net's parameters
        for param in self.net.parameters():
            param.requires_grad = False
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Computes the perceptual loss between input and target tensors.

        Args:
            input: Input tensor as ``torch.Tensor`` with shape [B, C, H, W].
            target: Target tensor as ``torch.Tensor`` with shape [B, C, H, W].

        Returns:
            Loss value as ``torch.Tensor``.
        """
        if self.preprocess:
            input  = self.run_preprocess(input)
            target = self.run_preprocess(target)
        input_feats  = self.get_features(input)
        target_feats = self.get_features(target)
        
        loss = 0
        for xf, yf in zip(input_feats, target_feats):
            loss += self.l1_loss(xf, yf)
        loss = loss / len(input_feats)
        return self.loss_weight * loss
    
    @staticmethod
    def run_preprocess(input: torch.Tensor) -> torch.Tensor:
        """Applies normalization preprocessing to the input tensor.

        Args:
            input: Input tensor as ``torch.Tensor`` with shape [B, C, H, W].

        Returns:
            Preprocessed tensor as ``torch.Tensor`` with shape [B, C, H, W].
        """
        transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        input     = transform(input)
        return input
    
    def get_features(self, input: torch.Tensor) -> list[torch.Tensor]:
        """Extracts features from specified layers of the network.

        Args:
            input: Input tensor as ``torch.Tensor`` with shape [B, C, H, W].

        Returns:
            List of feature tensors as ``list[torch.Tensor]``, shapes vary by layer.
        """
        x        = input
        features = []
        for name, layer in self.net._modules.items():
            x = layer(x)
            if name in self.layers:
                features.append(x)
        return features


@LOSSES.register(name="psnr_loss")
class PSNRLoss(base.Loss):
    """PSNR Loss computes the Peak Signal-to-Noise Ratio loss between input and target
    images.

    Args:
        to_y: Converts RGB to Y-channel (luminance) if ``True``. Default is ``False``.
        loss_weight: Weight of the loss as ``float``. Default is ``1.0``.
        reduction: Reduction method as ``Literal["none", "mean", "sum"]``.
            Default is ``"mean"``.

    References:
        - https://github.com/xinntao/BasicSR
    """
    
    def __init__(
        self,
        to_y       : bool  = False,
        loss_weight: float = 1.0,
        reduction  : Literal["none", "mean", "sum"] = "mean",
    ):
        super().__init__(loss_weight=loss_weight, reduction=reduction)
        self.scale = 10 / np.log(10)
        self.to_y  = to_y
        self.coef  = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Computes the PSNR loss between input and target tensors.

        Args:
            input: Input tensor as ``torch.Tensor`` with shape [B, C, H, W].
            target: Target tensor as ``torch.Tensor`` with shape [B, C, H, W].

        Returns:
            Loss value as ``torch.Tensor``.
        """
        x = input
        y = target
        
        if self.to_y:
            if self.first:
                self.coef = self.coef.to(x.device)
                self.first = False
            
            # Convert RGB to Y-channel (luminance) using ITU-R BT.601 coefficients
            x = (x * self.coef).sum(dim=1, keepdim=True) + 16.0  # [B, 1, H, W]
            y = (y * self.coef).sum(dim=1, keepdim=True) + 16.0  # [B, 1, H, W]
            x = x / 255.0
            y = y / 255.0
        
        # Compute Mean Squared Error (MSE) and PSNR
        mse  = torch.mean((x - y) ** 2, dim=[1, 2, 3])  # [B]
        psnr = -self.scale * torch.log10(mse + 1e-8)   # [B], negative PSNR as loss (higher PSNR = lower loss)
        
        # Apply reduction
        loss = base.reduce_loss(loss=psnr, reduction=self.reduction)
        return self.loss_weight * loss


@LOSSES.register(name="ssim_loss")
class SSIMLoss(base.Loss):
    """SSIM Loss computes the Structural Similarity Index Measure loss between input
    and target images.

    Args:
        data_range: Range of input data as ``float``. Default is ``255``.
        size_average: Average over each image if ``True``. Default is ``True``.
        window_size: Size of the Gaussian window as ``int``. Default is ``11``.
        window_sigma: Standard deviation of the Gaussian window as ``float``.
            Default is ``1.5``.
        channel: Number of channels in the input as ``int``. Default is ``3``.
        spatial_dims: Number of spatial dimensions as ``int``. Default is ``2``.
        k: Constants for SSIM calculation as ``tuple[float, float]`` (k1, k2).
            Default is ``(0.01, 0.03)``.
        non_negative_ssim: Ensures non-negative SSIM if ``True``. Default is ``False``.
        loss_weight: Weight of the loss as ``float``. Default is ``1.0``.
        reduction: Reduction method as ``Literal["none", "mean", "sum"]``.
            Default is ``"mean"``.
    """
    
    def __init__(
        self,
        data_range       : float = 255,
        size_average     : bool  = True,
        window_size      : int   = 11,
        window_sigma     : float = 1.5,
        channel          : int   = 3,
        spatial_dims     : int   = 2,
        k                : tuple[float, float] = (0.01, 0.03),
        non_negative_ssim: bool  = False,
        loss_weight      : float = 1.0,
        reduction        : Literal["none", "mean", "sum"] = "mean",
    ):
        super().__init__(loss_weight=loss_weight, reduction=reduction)
        from mon.nn.metric.pytorch_msssim import SSIM
        self.ssim = SSIM(
            data_range        = data_range,
            size_average      = size_average,
            window_size       = window_size,
            window_sigma      = window_sigma,
            channel           = channel,
            spatial_dims      = spatial_dims,
            k                 = k,
            non_negative_ssim = non_negative_ssim,
        )
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Computes the SSIM loss between input and target tensors.

        Args:
            input: Input tensor as ``torch.Tensor`` with shape [B, C, H, W].
            target: Target tensor as ``torch.Tensor`` with shape [B, C, H, W].

        Returns:
            Loss value as ``torch.Tensor``.
        """
        loss = 1.0 - self.ssim(input, target)
        return base.reduce_loss(loss=loss, reduction=self.reduction)


@LOSSES.register(name="ms_ssim_loss")
class MS_SSIMLoss(base.Loss):
    """MS-SSIM Loss computes the Multi-Scale Structural Similarity Index Measure loss between input and target images.

    Args:
        data_range: Range of input data as ``float``. Default is ``255``.
        size_average: Average over each image if ``True``. Default is ``True``.
        window_size: Size of the Gaussian window as ``int``. Default is ``11``.
        window_sigma: Standard deviation of the Gaussian window as ``float``.
            Default is ``1.5``.
        channel: Number of channels in the input as ``int``. Default is ``3``.
        spatial_dims: Number of spatial dimensions as ``int``. Default is ``2``.
        weights: Weights for each scale as ``list[float]`` or ``None``.
            Default is ``None``.
        k: Constants for SSIM calculation as ``tuple[float, float]`` (k1, k2).
            Default is ``(0.01, 0.03)``.
        loss_weight: Weight of the loss as ``float``. Default is ``1.0``.
        reduction: Reduction method as ``Literal["none", "mean", "sum"]``.
            Default is ``"mean"``.
    """
    
    def __init__(
        self,
        data_range  : float = 255,
        size_average: bool  = True,
        window_size : int   = 11,
        window_sigma: float = 1.5,
        channel     : int   = 3,
        spatial_dims: int   = 2,
        weights     : list[float] = None,
        k           : tuple[float, float] = (0.01, 0.03),
        loss_weight : float = 1.0,
        reduction   : Literal["none", "mean", "sum"] = "mean",
    ):
        super().__init__(loss_weight=loss_weight, reduction=reduction)
        from mon.nn.metric.pytorch_msssim import MS_SSIM
        self.ms_ssim = MS_SSIM(
            data_range   = data_range,
            size_average = size_average,
            window_size  = window_size,
            window_sigma = window_sigma,
            channel      = channel,
            spatial_dims = spatial_dims,
            weights      = weights,
            k            = k,
        )
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Computes the MS-SSIM loss between input and target tensors.

        Args:
            input: Input tensor as ``torch.Tensor`` with shape [B, C, H, W].
            target: Target tensor as ``torch.Tensor`` with shape [B, C, H, W].

        Returns:
            Loss value as ``torch.Tensor``.
        """
        loss = 1.0 - self.ms_ssim(input, target)
        return base.reduce_loss(loss=loss, reduction=self.reduction)

# endregion
