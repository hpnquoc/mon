#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Image Loss Functions.

This module implements loss functions for images.
"""

from __future__ import annotations

__all__ = [
    "ColorConstancyLoss",
    "DepthWeightedSmoothnessLoss",
    "EdgeAwareDepthConsistencyLoss",
    "EdgeAwareLoss",
    "EdgeLoss",
    "ExposureControlLoss",
    "ExposureValueControlLoss",
    "MSSSIMLoss",
    "PSNRLoss",
    "PerceptualLoss",
    "SSIMLoss",
    "SpatialConsistencyLoss",
    "TotalVariationLoss",
]

from typing import Literal

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.common_types import _size_2_t
from torchvision import models, transforms

from mon.globals import LOSSES
from mon.nn.loss import base


# region Color

@LOSSES.register(name="color_constancy_loss")
class ColorConstancyLoss(base.Loss):
    """Color Constancy Loss `\mathcal{L}_{col}` corrects the potential
    color deviations in the enhanced image and builds the relations among the
    three adjusted channels.
    
    References:
        https://github.com/Li-Chongyi/Zero-DCE/blob/master/Zero-DCE_code/Myloss.py#L9
    """
    
    def __init__(
        self,
        loss_weight: float = 1.0,
        reduction  : Literal["none", "mean", "sum"] = "mean"
    ):
        super().__init__(loss_weight=loss_weight, reduction=reduction)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        mean_rgb   = torch.mean(input, [2, 3], keepdim=True)
        mr, mg, mb = torch.split(mean_rgb, 1, dim=1)
        d_rg       = torch.pow(mr - mg, 2)
        d_rb       = torch.pow(mr - mb, 2)
        d_gb       = torch.pow(mb - mg, 2)
        loss       = torch.pow(torch.pow(d_rg, 2) + torch.pow(d_rb, 2) + torch.pow(d_gb, 2), 0.5)
        loss       = base.reduce_loss(loss=loss, reduction=self.reduction)
        loss       = self.loss_weight * loss
        return loss

# endregion


# region Depth

@LOSSES.register(name="depth_weighted_smoothness_loss")
class DepthWeightedSmoothnessLoss(base.Loss):
    """
    Calculate the depth-weighted smoothness loss for 4D tensors.
    
    Args:
        input: Predicted illumination map.
        depth: Depth map.
        alpha: Weighting factor for depth influence.
    """
    
    def __init__(
        self,
        alpha      : float = 1.0,
        loss_weight: float = 1.0,
        reduction  : Literal["none", "mean", "sum"] = "mean"
    ):
        super().__init__(loss_weight=loss_weight, reduction=reduction)
        self.alpha = alpha
    
    def forward(self, input: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        # Calculate gradients of illumination map (L) in x and y directions
        L_dx = input[:, :, :, 1:] - input[:, :, :, :-1]
        L_dy = input[:, :, 1:, :] - input[:, :, :-1, :]
        
        # Calculate gradients of depth map (D) in x and y directions
        D_dx = depth[:, :, :, 1:] - depth[:, :, :, :-1]
        D_dy = depth[:, :, 1:, :] - depth[:, :, :-1, :]
        
        # Compute depth-weighted terms for x and y directions
        weight_dx = torch.exp(-self.alpha * torch.abs(D_dx))
        weight_dy = torch.exp(-self.alpha * torch.abs(D_dy))
        
        # Apply depth weights to illumination gradients and take the mean
        loss_dx = torch.mean(weight_dx * torch.abs(L_dx))
        loss_dy = torch.mean(weight_dy * torch.abs(L_dy))
        
        # Sum the losses from both directions
        loss = loss_dx + loss_dy
        loss = self.loss_weight * loss
        return loss

# endregion


# region Edge

@LOSSES.register(name="edge_aware_loss")
class EdgeAwareLoss(base.Loss):
    
    def __init__(
        self,
        beta       : float = 1.0,
        loss_weight: float = 1.0,
        reduction  : Literal["none", "mean", "sum"] = "mean"
    ):
        super().__init__(loss_weight=loss_weight, reduction=reduction)
        self.beta = beta
    
    def forward(self, input: torch.Tensor, edge: torch.Tensor) -> torch.Tensor:
        # Calculate gradients of illumination map (L) in x and y directions
        L_dx = input[:, :, :, 1:] - input[:, :, :, :-1]
        L_dy = input[:, :, 1:, :] - input[:, :, :-1, :]
        
        # Calculate gradients of edge map (E) in x and y directions
        E_dx = edge[:, :, :, 1:] - edge[:, :, :, :-1]
        E_dy = edge[:, :, 1:, :] - edge[:, :, :-1, :]
        
        # Apply edge weights to illumination gradients; areas with stronger edges have lower weight
        # weight_dx = torch.exp(-torch.abs(E_dx))
        # weight_dy = torch.exp(-torch.abs(E_dy))
        weight_dx = 1 - self.beta * torch.abs(E_dx)
        weight_dy = 1 - self.beta * torch.abs(E_dy)
        
        # Calculate edge-aware losses by penalizing illumination changes along strong edges
        loss_dx = torch.mean(weight_dx * torch.abs(L_dx))
        loss_dy = torch.mean(weight_dy * torch.abs(L_dy))
        
        # Sum the losses from both directions
        loss = loss_dx + loss_dy
        loss = self.loss_weight * loss
        return loss
      
    
@LOSSES.register(name="edge_aware_depth_consistency_loss")
class EdgeAwareDepthConsistencyLoss(base.Loss):
    
    def __init__(
        self,
        tau        : float = 0.1,
        loss_weight: float = 1.0,
        reduction  : Literal["none", "mean", "sum"] = "mean"
    ):
        super().__init__(loss_weight=loss_weight, reduction=reduction)
        self.tau = tau
    
    def forward(self, input: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        # Compute depth edges
        depth_edges = self.compute_depth_edges(depth)
        # Apply a threshold to get edge-aware mask
        edge_mask   = (depth_edges > self.tau).float()  # Binary mask where edges are significant
        # Compute image gradients
        grad_pred_x, grad_pred_y = self.apply_sobel_filter_to_rgb(input)
        # Depth consistency loss between neighboring pixels
        loss = (edge_mask * (grad_pred_x ** 2 + grad_pred_y ** 2)).mean()
        loss = self.loss_weight * loss
        return loss
    
    # noinspection PyMethodMayBeStatic
    def compute_depth_edges(self, depth_map: torch.Tensor) -> torch.Tensor:
        sobel_kernel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [ 1,  0, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        sobel_kernel_y = torch.tensor([[1, 2,  1], [0, 0,  0], [-1, -2, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        sobel_kernel_x = sobel_kernel_x.to(depth_map.device)
        sobel_kernel_y = sobel_kernel_y.to(depth_map.device)
        grad_x         = F.conv2d(depth_map, sobel_kernel_x, padding=1)
        grad_y         = F.conv2d(depth_map, sobel_kernel_y, padding=1)
        # Compute magnitude of gradients
        grad_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2)
        return grad_magnitude
    
    # noinspection PyMethodMayBeStatic
    def apply_sobel_filter_to_rgb(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sobel_kernel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        sobel_kernel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        sobel_kernel_x = sobel_kernel_x.to(image.device)
        sobel_kernel_y = sobel_kernel_y.to(image.device)
        # Split the image into R, G, B channels
        channels = torch.chunk(image, chunks=3, dim=1)  # image shape [B, 3, H, W]
        # Apply Sobel filter to each channel
        grad_x_channels = [F.conv2d(channel, sobel_kernel_x, padding=1) for channel in channels]
        grad_y_channels = [F.conv2d(channel, sobel_kernel_y, padding=1) for channel in channels]
        # Stack the gradients back along the channel dimension
        grad_x = torch.cat(grad_x_channels, dim=1)
        grad_y = torch.cat(grad_y_channels, dim=1)
        return grad_x, grad_y


@LOSSES.register(name="edge_loss")
class EdgeLoss(base.Loss):
    
    def __init__(
        self,
        eps        : float = 1e-3,
        loss_weight: float = 1.0,
        reduction  : Literal["none", "mean", "sum"] = "mean"
    ):
        super().__init__(loss_weight=loss_weight, reduction=reduction)
        k           = torch.Tensor([[0.05, 0.25, 0.4, 0.25, 0.05]])
        self.kernel = torch.matmul(k.t(), k).unsqueeze(0).repeat(3, 1, 1, 1)
        self.loss   = base.CharbonnierLoss()

    def gauss_conv(self, image: torch.Tensor) -> torch.Tensor:
        b, c, w, h  = self.kernel.shape
        self.kernel = self.kernel.to(image.device)
        image       = F.pad(image, (w // 2, h // 2, w // 2, h // 2), mode="replicate")
        gauss       = F.conv2d(image, self.kernel, groups=b)
        return gauss
    
    def laplacian_kernel(self, image: torch.Tensor) -> torch.Tensor:
        filtered   = self.gauss_conv(image)       # filter
        down       = filtered[:, :, ::2, ::2]     # downsample
        new_filter = torch.zeros_like(filtered)
        new_filter[:, :, ::2, ::2] = down * 4     # upsample
        filtered   = self.gauss_conv(new_filter)  # filter
        diff       = image - filtered
        return diff
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        edge1 = self.laplacian_kernel(input)
        edge2 = self.laplacian_kernel(target)
        diff  = edge1 - edge2
        loss  = torch.mean(torch.sqrt((diff * diff) + (self.eps * self.eps)))
        loss  = base.reduce_loss(loss=loss, reduction=self.reduction)
        loss  = self.loss_weight * loss
        return loss

# endregion


# region Exposure

@LOSSES.register(name="exposure_control_loss")
class ExposureControlLoss(base.Loss):
    """Exposure Control Loss `\mathcal{L}_{exp}` measures the distance
    between the average intensity value of a local region to the
    well-exposedness level E.

    Args:
        patch_size: Kernel size for pooling layer.
        mean_val: The `E` value proposed in the paper. Default: ``0.6``.
        reduction: Specifies the reduction to apply to the output.
    
    References:
        https://github.com/Li-Chongyi/Zero-DCE/blob/master/Zero-DCE_code/Myloss.py#L74
    """
    
    def __init__(
        self,
        patch_size : _size_2_t = 16,
        mean_val   : float     = 0.6,
        loss_weight: float     = 1.0,
        reduction  : Literal["none", "mean", "sum"] = "mean",
    ):
        super().__init__(loss_weight=loss_weight, reduction=reduction)
        self.patch_size = patch_size
        self.mean_val   = mean_val
        self.pool       = nn.AvgPool2d(self.patch_size)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x    = input
        x    = torch.mean(x, 1, keepdim=True)
        mean = self.pool(x)
        loss = torch.pow(mean - torch.FloatTensor([self.mean_val]).to(input.device), 2)
        loss = base.reduce_loss(loss=loss, reduction=self.reduction)
        loss = self.loss_weight * loss
        return loss


@LOSSES.register(name="exposure_value_control_loss")
class ExposureValueControlLoss(base.Loss):
    """Exposure Value Control Loss measures the absolute value of
    `ExposureControlLoss`.

    Args:
        patch_size: Kernel size for pooling layer.
        mean_val: The `E` value (or optimally-intense threshold) proposed in the
            paper, lower values produce brighter images. Default: ``0.6``.
        reduction: Specifies the reduction to apply to the output.
    
    References:
        https://github.com/Li-Chongyi/Zero-DCE/blob/master/Zero-DCE_code/Myloss.py#L74
    """
    
    def __init__(
        self,
        patch_size : _size_2_t = 16,
        mean_val   : float     = 0.6,
        loss_weight: float     = 1.0,
        reduction  : Literal["none", "mean", "sum"] = "mean",
    ):
        super().__init__(loss_weight=loss_weight, reduction=reduction)
        self.patch_size = patch_size
        self.mean_val   = mean_val
        self.pool       = nn.AvgPool2d(self.patch_size)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x    = input
        x    = torch.mean(x, 1, keepdim=True)
        mean = self.pool(x) ** 0.5
        loss = torch.pow(mean - torch.FloatTensor([self.mean_val]).to(input.device), 2)
        loss = torch.abs(torch.mean(loss))
        loss = self.loss_weight * loss
        return loss

# endregion


# region Perceptual

@LOSSES.register(name="perceptual_loss")
class PerceptualLoss(base.Loss):
    """Perceptual Loss."""
    
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
        if self.preprocess:
            input  = self.run_preprocess(input)
            target = self.run_preprocess(target)
        input_feats  = self.get_features(input)
        target_feats = self.get_features(target)
        #
        loss = 0
        for xf, yf in zip(input_feats, target_feats):
            loss += self.l1_loss(xf, yf)
        loss = loss / len(input_feats)
        loss = self.loss_weight * loss
        return loss
    
    @staticmethod
    def run_preprocess(input: torch.Tensor) -> torch.Tensor:
        transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        input     = transform(input)
        return input
    
    def get_features(self, input: torch.Tensor) -> list[torch.Tensor]:
        x        = input
        features = []
        for name, layer in self.net._modules.items():
            x = layer(x)
            if name in self.layers:
                features.append(x)
        return features


@LOSSES.register(name="psnr_loss")
class PSNRLoss(base.Loss):
    """PSNR loss.
    
    References:
        https://github.com/xinntao/BasicSR
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
        assert len(input.size()) == 4
        if self.to_y:
            if self.first:
                self.coef  = self.coef.to(input.device)
                self.first = False
            
            input  = (input  * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.0
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.0
            input  = input  / 255.0
            target = target / 255.0
        
        psnr = torch.log(((input - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()
        # loss = reduce_loss(loss=loss, reduction=self.reduction)
        loss = self.loss_weight * self.scale * psnr
        return loss


@LOSSES.register(name="ssim_loss")
class SSIMLoss(base.Loss):
    
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
        loss = 1.0 - self.ssim(input, target)
        loss = base.reduce_loss(loss=loss, reduction=self.reduction)
        return loss


@LOSSES.register(name="ms_ssim_loss")
class MSSSIMLoss(base.Loss):

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
        loss = 1.0 - self.ms_ssim(input, target)
        loss = base.reduce_loss(loss=loss, reduction=self.reduction)
        return loss


@LOSSES.register(name="total_variation_loss")
class TotalVariationLoss(base.Loss):
    """Total Variation Loss on the Illumination (Illumination Smoothness Loss)
    `\mathcal{L}_{tvA}` preserve the monotonicity relations between
    neighboring pixels. It is used to avoid aggressive and sharp changes between
    neighboring pixels.
    
    References:
        `<https://github.com/Li-Chongyi/Zero-DCE/blob/master/Zero-DCE_code/Myloss.py>`__
    """
    
    def __init__(
        self,
        loss_weight: float = 1.0,
        reduction  : Literal["none", "mean", "sum"] = "mean",
    ):
        super().__init__(loss_weight=loss_weight, reduction=reduction)
    
    def forward(self, input : torch.Tensor) -> torch.Tensor:
        x       = input
        b       = x.size()[0]
        h_x     = x.size()[2]
        w_x     = x.size()[3]
        # count_h = (x.size()[2] - 1) * x.size()[3]
        # count_w = x.size()[2] * (x.size()[3] - 1)
        count_h = self._tensor_size(x[:, :, 1:, :])  # (x.size()[2]-1) * x.size()[3]
        count_w = self._tensor_size(x[:, :, :, 1:])  # x.size()[2] * (x.size()[3] - 1)
        h_tv    = torch.pow((x[:, :, 1:,  :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv    = torch.pow((x[:, :,  :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        loss    = 2 * (h_tv / count_h + w_tv / count_w) / b
        # loss    = base.reduce_loss(loss=loss, reduction=self.reduction)
        loss    = self.loss_weight * loss
        return loss
    
    @staticmethod
    def _tensor_size(t: torch.Tensor) -> int:
        return t.size()[1] * t.size()[2] * t.size()[3]

# endregion


# region Spatial

@LOSSES.register(name="spatial_consistency_loss")
class SpatialConsistencyLoss(base.Loss):
    """Spatial Consistency Loss `\mathcal{L}_{spa}` encourages spatial
    coherence of the enhanced image through preserving the difference of
    neighboring regions between the input image and its enhanced version.
    
    Args:
        num_regions: Number of neighboring regions. Default: ``4``.
        patch_size: The size of each neighboring region. Defaults: ``4`` means
            `4 x 4`.
    """
    
    def __init__(
        self,
        num_regions: Literal[4, 8, 16, 24] = 4,
        patch_size : int   = 4,
        loss_weight: float = 1.0,
        reduction  : Literal["none", "mean", "sum"] = "mean",
    ):
        super().__init__(loss_weight=loss_weight, reduction=reduction)
        self.num_regions = num_regions
        
        kernel_left = torch.FloatTensor([
            [ 0,  0, 0],
            [-1,  1, 0],
            [ 0,  0, 0]
        ]).unsqueeze(0).unsqueeze(0)
        kernel_right = torch.FloatTensor([
            [0,  0,  0],
            [0,  1, -1],
            [0,  0,  0]
        ]).unsqueeze(0).unsqueeze(0)
        kernel_up = torch.FloatTensor([
            [0, -1, 0],
            [0,  1, 0],
            [0,  0, 0]
        ]).unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor([
            [0,  0, 0],
            [0,  1, 0],
            [0, -1, 0]
        ]).unsqueeze(0).unsqueeze(0)
        if self.num_regions in [8, 16]:
            kernel_upleft = torch.FloatTensor([
                [-1, 0, 0],
                [ 0, 1, 0],
                [ 0, 0, 0]
            ]).unsqueeze(0).unsqueeze(0)
            kernel_upright = torch.FloatTensor([
                [0, 0, -1],
                [0, 1,  0],
                [0, 0,  0]
            ]).unsqueeze(0).unsqueeze(0)
            kernel_downleft = torch.FloatTensor([
                [ 0, 0, 0],
                [ 0, 1, 0],
                [-1, 0, 0]
            ]).unsqueeze(0).unsqueeze(0)
            kernel_downright = torch.FloatTensor([
                [0, 0,  0],
                [0, 1,  0],
                [0, 0, -1]
            ]).unsqueeze(0).unsqueeze(0)
        if self.num_regions in [16, 24]:
            kernel_left2 = torch.FloatTensor([
                [0,  0,  0, 0, 0],
                [0,  0,  0, 0, 0],
                [-1, 0,  1, 0, 0],
                [0,  0,  0, 0, 0],
                [0,  0,  0, 0, 0]
            ]).unsqueeze(0).unsqueeze(0)
            kernel_right2 = torch.FloatTensor([
                [0, 0,  0, 0,  0],
                [0, 0,  0, 0,  0],
                [0, 0,  1, 0, -1],
                [0, 0,  0, 0,  0],
                [0, 0,  0, 0,  0]
            ]).unsqueeze(0).unsqueeze(0)
            kernel_up2 = torch.FloatTensor([
                [0, 0, -1, 0, 0],
                [0, 0,  0, 0, 0],
                [0, 0,  1, 0, 0],
                [0, 0,  0, 0, 0],
                [0, 0,  0, 0, 0]
            ]).unsqueeze(0).unsqueeze(0)
            kernel_down2 = torch.FloatTensor([
                [0, 0,  0, 0, 0],
                [0, 0,  0, 0, 0],
                [0, 0,  1, 0, 0],
                [0, 0,  0, 0, 0],
                [0, 0, -1, 0, 0]
            ]).unsqueeze(0).unsqueeze(0)
            kernel_up2left2 = torch.FloatTensor([
                [-1, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0],
                [ 0, 0, 1, 0, 0],
                [ 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0]
            ]).unsqueeze(0).unsqueeze(0)
            kernel_up2right2 = torch.FloatTensor([
                [0, 0, 0, 0, -1],
                [0, 0, 0, 0,  0],
                [0, 0, 1, 0,  0],
                [0, 0, 0, 0,  0],
                [0, 0, 0, 0,  0]
            ]).unsqueeze(0).unsqueeze(0)
            kernel_down2left2 = torch.FloatTensor([
                [ 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0],
                [ 0, 0, 1, 0, 0],
                [ 0, 0, 0, 0, 0],
                [-1, 0, 0, 0, 0]
            ]).unsqueeze(0).unsqueeze(0)
            kernel_down2right2 = torch.FloatTensor([
                [0, 0, 0, 0,  0],
                [0, 0, 0, 0,  0],
                [0, 0, 1, 0,  0],
                [0, 0, 0, 0,  0],
                [0, 0, 0, 0, -1]
            ]).unsqueeze(0).unsqueeze(0)
        if self.num_regions in [24]:
            kernel_up2left1 = torch.FloatTensor([
                [0, -1, 0, 0, 0],
                [0,  0, 0, 0, 0],
                [0,  0, 1, 0, 0],
                [0,  0, 0, 0, 0],
                [0,  0, 0, 0, 0]
            ]).unsqueeze(0).unsqueeze(0)
            kernel_up2right1 = torch.FloatTensor([
                [0, 0, 0, -1, 0],
                [0, 0, 0,  0, 0],
                [0, 0, 1,  0, 0],
                [0, 0, 0,  0, 0],
                [0, 0, 0,  0, 0]
            ]).unsqueeze(0).unsqueeze(0)
            kernel_up1left2 = torch.FloatTensor([
                [0,  0, 0, 0, 0],
                [-1, 0, 0, 0, 0],
                [0,  0, 1, 0, 0],
                [0,  0, 0, 0, 0],
                [0,  0, 0, 0, 0]
            ]).unsqueeze(0).unsqueeze(0)
            kernel_up1right2 = torch.FloatTensor([
                [0, 0, 0, 0,  0],
                [0, 0, 0, 0, -1],
                [0, 0, 1, 0,  0],
                [0, 0, 0, 0,  0],
                [0, 0, 0, 0,  0]
            ]).unsqueeze(0).unsqueeze(0)
            kernel_down2left1 = torch.FloatTensor([
                [0,  0, 0, 0, 0],
                [0,  0, 0, 0, 0],
                [0,  0, 1, 0, 0],
                [0,  0, 0, 0, 0],
                [0, -1, 0, 0, 0]
            ]).unsqueeze(0).unsqueeze(0)
            kernel_down2right1 = torch.FloatTensor([
                [0, 0, 0,  0, 0],
                [0, 0, 0,  0, 0],
                [0, 0, 1,  0, 0],
                [0, 0, 0,  0, 0],
                [0, 0, 0, -1, 0]
            ]).unsqueeze(0).unsqueeze(0)
            kernel_down1left2 = torch.FloatTensor([
                [ 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0],
                [ 0, 0, 1, 0, 0],
                [-1, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0]
            ]).unsqueeze(0).unsqueeze(0)
            kernel_down1right2 = torch.FloatTensor([
                [0, 0, 0, 0,  0],
                [0, 0, 0, 0,  0],
                [0, 0, 1, 0,  0],
                [0, 0, 0, 0, -1],
                [0, 0, 0, 0,  0]
            ]).unsqueeze(0).unsqueeze(0)
            
        self.weight_left  = nn.Parameter(data=kernel_left,  requires_grad=False)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_up    = nn.Parameter(data=kernel_up,    requires_grad=False)
        self.weight_down  = nn.Parameter(data=kernel_down,  requires_grad=False)
        if self.num_regions in [8, 16]:
            self.weight_upleft    = nn.Parameter(data=kernel_upleft,    requires_grad=False)
            self.weight_upright   = nn.Parameter(data=kernel_upright,   requires_grad=False)
            self.weight_downleft  = nn.Parameter(data=kernel_downleft,  requires_grad=False)
            self.weight_downright = nn.Parameter(data=kernel_downright, requires_grad=False)
        if self.num_regions in [16, 24]:
            self.weight_left2       = nn.Parameter(data=kernel_left2,       requires_grad=False)
            self.weight_right2      = nn.Parameter(data=kernel_right2,      requires_grad=False)
            self.weight_up2         = nn.Parameter(data=kernel_up2,         requires_grad=False)
            self.weight_down2       = nn.Parameter(data=kernel_down2,       requires_grad=False)
            self.weight_up2left2    = nn.Parameter(data=kernel_up2left2,    requires_grad=False)
            self.weight_up2right2   = nn.Parameter(data=kernel_up2right2,   requires_grad=False)
            self.weight_down2left2  = nn.Parameter(data=kernel_down2left2,  requires_grad=False)
            self.weight_down2right2 = nn.Parameter(data=kernel_down2right2, requires_grad=False)
        if self.num_regions in [24]:
            self.weight_up2left1    = nn.Parameter(data=kernel_up2left1,    requires_grad=False)
            self.weight_up2right1   = nn.Parameter(data=kernel_up2right1,   requires_grad=False)
            self.weight_up1left2    = nn.Parameter(data=kernel_up1left2,    requires_grad=False)
            self.weight_up1right2   = nn.Parameter(data=kernel_up1right2,   requires_grad=False)
            self.weight_down2left1  = nn.Parameter(data=kernel_down2left1,  requires_grad=False)
            self.weight_down2right1 = nn.Parameter(data=kernel_down2right1, requires_grad=False)
            self.weight_down1left2  = nn.Parameter(data=kernel_down1left2,  requires_grad=False)
            self.weight_down1right2 = nn.Parameter(data=kernel_down1right2, requires_grad=False)
        
        self.pool = nn.AvgPool2d(patch_size)  # Default 4
    
    def __str__(self) -> str:
        return f"spatial_consistency_loss"
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.weight_left.device != input.device:
            self.weight_left = self.weight_left.to(input.device)
        if self.weight_right.device != input.device:
            self.weight_right = self.weight_right.to(input.device)
        if self.weight_up.device != input.device:
            self.weight_up = self.weight_up.to(input.device)
        if self.weight_down.device != input.device:
            self.weight_down = self.weight_down.to(input.device)
        if self.num_regions in [8, 16]:
            if self.weight_upleft.device != input.device:
                self.weight_upleft = self.weight_upleft.to(input.device)
            if self.weight_upright.device != input.device:
                self.weight_upright = self.weight_upright.to(input.device)
            if self.weight_downleft.device != input.device:
                self.weight_downleft = self.weight_downleft.to(input.device)
            if self.weight_downright.device != input.device:
                self.weight_downright = self.weight_downright.to(input.device)
        if self.num_regions in [16, 24]:
            if self.weight_left2.device != input.device:
                self.weight_left2 = self.weight_left2.to(input.device)
            if self.weight_right2.device != input.device:
                self.weight_right2 = self.weight_right2.to(input.device)
            if self.weight_up2.device != input.device:
                self.weight_up2 = self.weight_up2.to(input.device)
            if self.weight_down2.device != input.device:
                self.weight_down2 = self.weight_down2.to(input.device)
            if self.weight_up2left2.device != input.device:
                self.weight_up2left2 = self.weight_up2left2.to(input.device)
            if self.weight_up2right2.device != input.device:
                self.weight_up2right2 = self.weight_up2right2.to(input.device)
            if self.weight_down2left2.device != input.device:
                self.weight_down2left2 = self.weight_down2left2.to(input.device)
            if self.weight_down2right2.device != input.device:
                self.weight_down2right2 = self.weight_down2right2.to(input.device)
        if self.num_regions in [24]:
            if self.weight_up2left1.device != input.device:
                self.weight_up2left1 = self.weight_up2left1.to(input.device)
            if self.weight_up2right1.device != input.device:
                self.weight_up2right1 = self.weight_up2right1.to(input.device)
            if self.weight_up1left2.device != input.device:
                self.weight_up1left2 = self.weight_up1left2.to(input.device)
            if self.weight_up1right2.device != input.device:
                self.weight_up1right2 = self.weight_up1right2.to(input.device)
            if self.weight_down2left1.device != input.device:
                self.weight_down2left1 = self.weight_down2left1.to(input.device)
            if self.weight_down2right1.device != input.device:
                self.weight_down2right1 = self.weight_down2right1.to(input.device)
            if self.weight_down1left2.device != input.device:
                self.weight_down1left2 = self.weight_down1left2.to(input.device)
            if self.weight_down1right2.device != input.device:
                self.weight_down1right2 = self.weight_down1right2.to(input.device)
                
        org_mean     = torch.mean(input,  1, keepdim=True)
        enhance_mean = torch.mean(target, 1, keepdim=True)
        
        org_pool     = self.pool(org_mean)
        enhance_pool = self.pool(enhance_mean)
        
        d_org_left   = F.conv2d(org_pool, self.weight_left,  padding=1)
        d_org_right  = F.conv2d(org_pool, self.weight_right, padding=1)
        d_org_up     = F.conv2d(org_pool, self.weight_up,    padding=1)
        d_org_down   = F.conv2d(org_pool, self.weight_down,  padding=1)
        if self.num_regions in [8, 16]:
            d_org_upleft    = F.conv2d(org_pool, self.weight_upleft,    padding=1)
            d_org_upright   = F.conv2d(org_pool, self.weight_upright,   padding=1)
            d_org_downleft  = F.conv2d(org_pool, self.weight_downleft,  padding=1)
            d_org_downright = F.conv2d(org_pool, self.weight_downright, padding=1)
        if self.num_regions in [16, 24]:
            d_org_left2       = F.conv2d(org_pool, self.weight_left2,       padding=2)
            d_org_right2      = F.conv2d(org_pool, self.weight_right2,      padding=2)
            d_org_up2         = F.conv2d(org_pool, self.weight_up2,         padding=2)
            d_org_down2       = F.conv2d(org_pool, self.weight_down2,       padding=2)
            d_org_up2left2    = F.conv2d(org_pool, self.weight_up2left2,    padding=2)
            d_org_up2right2   = F.conv2d(org_pool, self.weight_up2right2,   padding=2)
            d_org_down2left2  = F.conv2d(org_pool, self.weight_down2left2,  padding=2)
            d_org_down2right2 = F.conv2d(org_pool, self.weight_down2right2, padding=2)
        if self.num_regions in [24]:
            d_org_up2left1    = F.conv2d(org_pool, self.weight_up2left1,    padding=2)
            d_org_up2right1   = F.conv2d(org_pool, self.weight_up2right1,   padding=2)
            d_org_up1left2    = F.conv2d(org_pool, self.weight_up1left2,    padding=2)
            d_org_up1right2   = F.conv2d(org_pool, self.weight_up1right2,   padding=2)
            d_org_down2left1  = F.conv2d(org_pool, self.weight_down2left1,  padding=2)
            d_org_down2right1 = F.conv2d(org_pool, self.weight_down2right1, padding=2)
            d_org_down1left2  = F.conv2d(org_pool, self.weight_down1left2,  padding=2)
            d_org_down1right2 = F.conv2d(org_pool, self.weight_down1right2, padding=2)
        
        d_enhance_left  = F.conv2d(enhance_pool, self.weight_left,  padding=1)
        d_enhance_right = F.conv2d(enhance_pool, self.weight_right, padding=1)
        d_enhance_up    = F.conv2d(enhance_pool, self.weight_up,    padding=1)
        d_enhance_down  = F.conv2d(enhance_pool, self.weight_down,  padding=1)
        if self.num_regions in [8, 16]:
            d_enhance_upleft    = F.conv2d(enhance_pool, self.weight_upleft,    padding=1)
            d_enhance_upright   = F.conv2d(enhance_pool, self.weight_upright,   padding=1)
            d_enhance_downleft  = F.conv2d(enhance_pool, self.weight_downleft,  padding=1)
            d_enhance_downright = F.conv2d(enhance_pool, self.weight_downright, padding=1)
        if self.num_regions in [16, 24]:
            d_enhance_left2       = F.conv2d(enhance_pool, self.weight_left2,       padding=2)
            d_enhance_right2      = F.conv2d(enhance_pool, self.weight_right2,      padding=2)
            d_enhance_up2         = F.conv2d(enhance_pool, self.weight_up2,         padding=2)
            d_enhance_down2       = F.conv2d(enhance_pool, self.weight_down2,       padding=2)
            d_enhance_up2left2    = F.conv2d(enhance_pool, self.weight_up2left2,    padding=2)
            d_enhance_up2right2   = F.conv2d(enhance_pool, self.weight_up2right2,   padding=2)
            d_enhance_down2left2  = F.conv2d(enhance_pool, self.weight_down2left2,  padding=2)
            d_enhance_down2right2 = F.conv2d(enhance_pool, self.weight_down2right2, padding=2)
        if self.num_regions in [24]:
            d_enhance_up2left1    = F.conv2d(enhance_pool, self.weight_up2left1,    padding=2)
            d_enhance_up2right1   = F.conv2d(enhance_pool, self.weight_up2right1,   padding=2)
            d_enhance_up1left2    = F.conv2d(enhance_pool, self.weight_up1left2,    padding=2)
            d_enhance_up1right2   = F.conv2d(enhance_pool, self.weight_up1right2,   padding=2)
            d_enhance_down2left1  = F.conv2d(enhance_pool, self.weight_down2left1,  padding=2)
            d_enhance_down2right1 = F.conv2d(enhance_pool, self.weight_down2right1, padding=2)
            d_enhance_down1left2  = F.conv2d(enhance_pool, self.weight_down1left2,  padding=2)
            d_enhance_down1right2 = F.conv2d(enhance_pool, self.weight_down1right2, padding=2)
        
        d_left  = torch.pow(d_org_left  - d_enhance_left,  2)
        d_right = torch.pow(d_org_right - d_enhance_right, 2)
        d_up    = torch.pow(d_org_up    - d_enhance_up,    2)
        d_down  = torch.pow(d_org_down  - d_enhance_down,  2)
        if self.num_regions in [8, 16]:
            d_upleft    = torch.pow(d_org_upleft    - d_enhance_upleft,    2)
            d_upright   = torch.pow(d_org_upright   - d_enhance_upright,   2)
            d_downleft  = torch.pow(d_org_downleft  - d_enhance_downleft,  2)
            d_downright = torch.pow(d_org_downright - d_enhance_downright, 2)
        if self.num_regions in [16, 24]:
            d_left2       = torch.pow(d_org_left2       - d_enhance_left2      , 2)
            d_right2      = torch.pow(d_org_right2      - d_enhance_right2     , 2)
            d_up2         = torch.pow(d_org_up2         - d_enhance_up2        , 2)
            d_down2       = torch.pow(d_org_down2       - d_enhance_down2      , 2)
            d_up2left2    = torch.pow(d_org_up2left2    - d_enhance_up2left2   , 2)
            d_up2right2   = torch.pow(d_org_up2right2   - d_enhance_up2right2  , 2)
            d_down2left2  = torch.pow(d_org_down2left2  - d_enhance_down2left2 , 2)
            d_down2right2 = torch.pow(d_org_down2right2 - d_enhance_down2right2, 2)
        if self.num_regions in [24]:
            d_up2left1    = torch.pow(d_org_up2left1    - d_enhance_up2left1   , 2)
            d_up2right1   = torch.pow(d_org_up2right1   - d_enhance_up2right1  , 2)
            d_up1left2    = torch.pow(d_org_up1left2    - d_enhance_up1left2   , 2)
            d_up1right2   = torch.pow(d_org_up1right2   - d_enhance_up1right2  , 2)
            d_down2left1  = torch.pow(d_org_down2left1  - d_enhance_down2left1 , 2)
            d_down2right1 = torch.pow(d_org_down2right1 - d_enhance_down2right1, 2)
            d_down1left2  = torch.pow(d_org_down1left2  - d_enhance_down1left2 , 2)
            d_down1right2 = torch.pow(d_org_down1right2 - d_enhance_down1right2, 2)
        
        loss = d_left + d_right + d_up + d_down
        if self.num_regions in [8, 16]:
            loss += d_upleft + d_upright + d_downleft + d_downright
        if self.num_regions in [16, 24]:
            loss += (d_left2 + d_right2 + d_up2 + d_down2 +
                     d_up2left2 + d_up2right2 + d_down2left2 + d_down2right2)
        if self.num_regions in [24]:
            loss += (d_up2left1 + d_up2right1 + d_up1left2 + d_up1right2 +
                     d_down2left1 + d_down2right1 + d_down1left2 + d_down1right2)
        
        loss = base.reduce_loss(loss=loss, reduction=self.reduction)
        loss = self.loss_weight * loss
        return loss

# endregion
