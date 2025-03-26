#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Attention Layers.

This module implements attention layers.
"""

from __future__ import annotations

__all__ = [
    "BAM",
    "CBAM",
    "ChannelAttention",
    "ChannelAttentionModule",
    "ECA",
    "ECA1d",
    "EfficientChannelAttention",
    "EfficientChannelAttention1d",
    "GalerkinSimpleAttention",
    "PAM",
    "PixelAttentionModule",
    "SimAM",
    "SimplifiedChannelAttention",
    "SqueezeExcitation",
    "SqueezeExciteC",
    "SqueezeExciteL",
]

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.common_types import _size_2_t
from torchvision.ops.misc import SqueezeExcitation


# region Channel Attention

class EfficientChannelAttention(nn.Module):
    """Efficient Channel Attention (ECA) module.

    Args:
        channels: Number of input channels.
        kernel_size: Kernel size for 1D convolution. Default is ``3``.
    """

    def __init__(self, channels: int, kernel_size: _size_2_t = 3):
        super().__init__()
        padding       = (kernel_size - 1) // 2
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv     = nn.Conv1d(
            in_channels  = 1,
            out_channels = 1,
            kernel_size  = kernel_size,
            padding      = padding,
            bias         = False
        )
        self.sigmoid  = nn.Sigmoid()
        self.channel  = channels
        self.k_size   = kernel_size

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Applies efficient channel attention.

        Args:
            input: Input tensor [B, C, H, W].

        Returns:
            Output tensor [B, C, H, W] with channel attention applied.
        """
        x = input
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

    def flops(self) -> int:
        """Calculates FLOPs for the module.

        Returns:
            Number of floating-point operations.
        """
        return self.channel * self.channel * self.k_size


class EfficientChannelAttention1d(nn.Module):
    """Efficient Channel Attention (ECA) module for 1D inputs.

    Args:
        channels: Number of input channels.
        kernel_size: Kernel size for 1D convolution. Default is ``3``.
    """

    def __init__(
        self,
        channels   : int,
        kernel_size: _size_2_t = 3
    ):
        super().__init__()
        padding       = (kernel_size - 1) // 2
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv     = nn.Conv1d(
            in_channels  = 1,
            out_channels = 1,
            kernel_size  = kernel_size,
            padding      = padding,
            bias         = False
        )
        self.sigmoid  = nn.Sigmoid()
        self.channel  = channels
        self.k_size   = kernel_size

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Applies efficient channel attention to 1D input.

        Args:
            input: Input tensor ``[B, C, L]``.

        Returns:
            Output tensor ``[B, C, L]`` with channel attention applied.
        """
        x = input
        y = self.avg_pool(x)                   # [B, C, 1]
        y = self.conv(y.transpose(-1, -2))     # [B, 1, C] -> [B, 1, C]
        y = self.sigmoid(y.transpose(-1, -2))  # [B, C, 1]
        return x * y.expand_as(x)

    def flops(self) -> int:
        """Calculates FLOPs for the module.

        Returns:
            Number of floating-point operations.
        """
        return self.channel * self.channel * self.k_size


class SimplifiedChannelAttention(nn.Module):
    """Simplified channel attention from 'Simple Baselines for Image Restoration'.

    Args:
        channels: Number of input/output channels.
        bias: If ``True``, adds bias to convolution. Default is ``True``.
        device: Device for the module. Default is ``None``.
        dtype: Data type for the module. Default is ``None``.

    References:
        https://arxiv.org/pdf/2204.04676.pdf
    """

    def __init__(
        self,
        channels: int,
        bias    : bool = True,
        device  : Any  = None,
        dtype   : Any  = None
    ):
        super().__init__()
        self.avg_pool   = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Conv2d(
            in_channels  = channels,
            out_channels = channels,
            kernel_size  = 1,
            stride       = 1,
            padding      = 0,
            bias         = bias,
            device       = device,
            dtype        = dtype
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Applies simplified channel attention.

        Args:
            input: Input tensor [B, C, H, W].

        Returns:
            Output tensor [B, C, H, W] with channel attention applied.
        """
        x = input
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)  # [B, C, 1, 1] -> [B, C]
        y = self.excitation(y.view(b, c, 1, 1)).view(b, c, 1, 1)  # [B, C] -> [B, C, 1, 1]
        return x * y.expand_as(x)


ECA   = EfficientChannelAttention
ECA1d = EfficientChannelAttention1d

# endregion


# region Channel Attention Module

class BAM(nn.Module):
    """Bottleneck Attention Module from BAM paper.

    References:
        https://github.com/Jongchan/attention-module/blob/master/MODELS/bam.py
    """

    class Flatten(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            """Flattens input tensor to ``[B, -1]``.

            Args:
                input: Input tensor of any shape.

            Returns:
                Flattened tensor ``[B, -1]``.
            """
            return input.view(input.size(0), -1)

    class ChannelAttention(nn.Module):
        def __init__(
            self,
            channels       : int,
            reduction_ratio: int = 16,
            num_layers     : int = 1
        ):
            """Initializes the ChannelAttention module.

            Args:
                channels: Number of input channels.
                reduction_ratio: Channel reduction ratio. Default is ``16``.
                num_layers: Number of hidden layers. Default is ``1``.
            """
            super().__init__()
            gate_channels = [channels] + [channels // reduction_ratio] * num_layers + [channels]
            self.c_gate = nn.Sequential(
                BAM.Flatten(),
                *[
                    nn.Sequential(
                        nn.Linear(gate_channels[i], gate_channels[i + 1]),
                        nn.BatchNorm1d(gate_channels[i + 1]),
                        nn.ReLU()
                    ) for i in range(len(gate_channels) - 2)
                ],
                nn.Linear(gate_channels[-2], gate_channels[-1])
            )

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            """Applies channel attention.

            Args:
                input: Input tensor [B, C, H, W].

            Returns:
                Attention weights tensor ``[B, C, 1, 1]``.
            """
            y = F.avg_pool2d(input, input.size()[2:], stride=input.size()[2:])  # [B, C, 1, 1]
            y = self.c_gate(y)  # [B, C]
            return y.unsqueeze(2).unsqueeze(3).expand_as(input)  # [B, C, H, W]

    class SpatialAttention(nn.Module):
        def __init__(
            self,
            channels         : int,
            reduction_ratio  : int = 16,
            dilation_conv_num: int = 2,
            dilation_val     : int = 4
        ):
            """Initializes the SpatialAttention module.

            Args:
                channels: Number of input channels.
                reduction_ratio: Channel reduction ratio. Default is ``16``.
                dilation_conv_num: Number of dilated convolutions. Default is ``2``.
                dilation_val: Dilation value for convolutions. Default is ``4``.
            """
            super().__init__()
            self.s_gate = nn.Sequential(
                nn.Conv2d(
                    in_channels  = channels,
                    out_channels = channels // reduction_ratio,
                    kernel_size  = 1
                ),
                nn.BatchNorm2d(channels // reduction_ratio),
                nn.ReLU(),
                *[
                    nn.Sequential(
                        nn.Conv2d(
                            in_channels  = channels // reduction_ratio,
                            out_channels = channels // reduction_ratio,
                            kernel_size  = 3,
                            padding      = dilation_val,
                            dilation     = dilation_val
                        ),
                        nn.BatchNorm2d(channels // reduction_ratio),
                        nn.ReLU()
                    ) for _ in range(dilation_conv_num)
                ],
                nn.Conv2d(
                    in_channels  = channels // reduction_ratio,
                    out_channels = 1,
                    kernel_size  = 1
                )
            )

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            """Applies spatial attention.

            Args:
                input: Input tensor [B, C, H, W].

            Returns:
                Attention weights tensor ``[B, 1, H, W]``.
            """
            return self.s_gate(input).expand_as(input)

    def __init__(
        self,
        channels       : int,
        reduction_ratio: int = 16,
        num_layers     : int = 1
    ):
        """Initializes the BAM module.

        Args:
            channels: Number of input channels.
            reduction_ratio: Channel reduction ratio. Default is ``16``.
            num_layers: Number of hidden layers in channel attention. Default is ``1``.
        """
        super().__init__()
        self.channel_att = self.ChannelAttention(
            channels        = channels,
            reduction_ratio = reduction_ratio,
            num_layers      = num_layers
        )
        self.spatial_att = self.SpatialAttention(
            channels        = channels,
            reduction_ratio = reduction_ratio
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Applies bottleneck attention.

        Args:
            input: Input tensor [B, C, H, W].

        Returns:
            Output tensor [B, C, H, W] with attention applied.
        """
        y = 1 + self.sigmoid(self.channel_att(input) * self.spatial_att(input))
        return input * y


class CBAM(nn.Module):
    """Convolutional Block Attention Module from CBAM paper.

    Args:
        channels: Number of input channels.
        reduction_ratio: Channel reduction ratio. Default is ``16``.
        pool_types: Pooling layer types. Default is ``["avg", "max"]``.
        spatial: If ``True``, includes spatial attention. Default is ``True``.

    References:
        https://github.com/Jongchan/attention-module/blob/master/MODELS/cbam.py
    """

    class Flatten(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            """Flattens input tensor to ``[B, -1]``.

            Args:
                input: Input tensor of any shape.

            Returns:
                Flattened tensor ``[B, -1]``.
            """
            return input.view(input.size(0), -1)

    class ChannelAttention(nn.Module):
        def __init__(
            self,
            channels       : int,
            reduction_ratio: int = 16,
            pool_types     : list[str] = ["avg", "max"]
        ):
            """Initializes the ``ChannelAttention`` module.

            Args:
                channels: Number of input channels.
                reduction_ratio: Channel reduction ratio. Default is ``16``.
                pool_types: Pooling layer types. Default is ``["avg", "max"]``.
            """
            super().__init__()
            self.mlp = nn.Sequential(
                CBAM.Flatten(),
                nn.Linear(channels, channels // reduction_ratio),
                nn.ReLU(),
                nn.Linear(channels // reduction_ratio, channels)
            )
            self.pool_types = pool_types

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            """Applies channel attention.

            Args:
                input: Input tensor [B, C, H, W].

            Returns:
                Output tensor [B, C, H, W] with channel attention applied.
            """
            channel_att_sum = sum(
                self.mlp(
                    getattr(F, f"{pool_type}_pool2d")(
                        input, input.size()[2:], stride=input.size()[2:]
                    )
                ) for pool_type in self.pool_types
            )
            return input * torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(input)

    class SpatialAttention(nn.Module):
        def __init__(self):
            """Initializes the ``SpatialAttention`` module."""
            super().__init__()
            self.spatial = nn.Sequential(
                nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
                nn.BatchNorm2d(1),
                nn.Sigmoid()
            )

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            """Applies spatial attention.

            Args:
                input: Input tensor [B, C, H, W].

            Returns:
                Output tensor [B, C, H, W] with spatial attention applied.
            """
            y = torch.cat([input.mean(dim=1, keepdim=True), input.max(dim=1, keepdim=True)[0]], dim=1)
            return input * self.spatial(y).expand_as(input)

    def __init__(
        self,
        channels       : int,
        reduction_ratio: int       = 16,
        pool_types     : list[str] = ["avg", "max"],
        spatial        : bool      = True
    ):
        """Initializes the ``CBAM`` module.

        Args:
            channels: Number of input channels.
            reduction_ratio: Channel reduction ratio. Default is ``16``.
            pool_types: Pooling layer types. Default is ``["avg", "max"]``.
            spatial: If ``True``, includes spatial attention. Default is ``True``.
        """
        super().__init__()
        self.channel_att = self.ChannelAttention(
            channels        = channels,
            reduction_ratio = reduction_ratio,
            pool_types      = pool_types
        )
        self.spatial_att = self.SpatialAttention() if spatial else nn.Identity()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Applies convolutional block attention.

        Args:
            input: Input tensor [B, C, H, W].

        Returns:
            Output tensor [B, C, H, W] with attention applied.
        """
        return self.spatial_att(self.channel_att(input))


class ChannelAttentionModule(nn.Module):
    """Channel Attention Module for feature enhancement.

    Args:
        channels: Number of input channels.
        reduction_ratio: Channel reduction ratio.
        stride: Stride of the first convolution. Default is ``1``.
        padding: Padding of the first convolution. Default is ``0``.
        dilation: Dilation of the convolutions. Default is ``1``.
        groups: Number of groups in the convolutions. Default is ``1``.
        bias: If ``True``, adds bias to convolutions. Default is ``True``.
        padding_mode: Padding mode for convolutions. Default is ``"zeros"``.
        device: Device for the module. Default is ``None``.
        dtype: Data type for the module. Default is ``None``.
    """

    def __init__(
        self,
        channels       : int,
        reduction_ratio: int,
        stride         : int  = 1,
        padding        : int  = 0,
        dilation       : int  = 1,
        groups         : int  = 1,
        bias           : bool = True,
        padding_mode   : str  = "zeros",
        device         : Any  = None,
        dtype          : Any  = None
    ):
        super().__init__()
        self.avg_pool   = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Conv2d(
                in_channels  = channels,
                out_channels = channels // reduction_ratio,
                kernel_size  = 1,
                stride       = stride,
                padding      = padding,
                dilation     = dilation,
                groups       = groups,
                bias         = bias,
                padding_mode = padding_mode,
                device       = device,
                dtype        = dtype
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels  = channels // reduction_ratio,
                out_channels = channels,
                kernel_size  = 1,
                padding      = 0,
                dilation     = dilation,
                groups       = groups,
                bias         = bias,
                padding_mode = padding_mode,
                device       = device,
                dtype        = dtype
            ),
            nn.Sigmoid()
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Applies channel attention to the input.

        Args:
            input: Input tensor [B, C, H, W].

        Returns:
            Output tensor [B, C, H, W] with channel attention applied.
        """
        return input * self.excitation(self.avg_pool(input))

# endregion


# region Galerkin-type Attention

class GalerkinSimpleAttention(nn.Module):
    """Galerkin-type attention mechanism.

    Args:
        mid_channels: Number of intermediate channels.
        heads: Number of attention heads.

    References:
        https://github.com/2y7c3/Super-Resolution-Neural-Operator/blob/main/models/galerkin.py
    """

    def __init__(self, mid_channels: int, heads: int):
        super().__init__()
        self.headc = mid_channels // heads
        self.heads = heads

        self.qkv_proj = nn.Conv2d(mid_channels, 3 * mid_channels, 1)
        self.o_proj1  = nn.Conv2d(mid_channels, mid_channels, 1)
        self.o_proj2  = nn.Conv2d(mid_channels, mid_channels, 1)

        self.kln = nn.LayerNorm((heads, 1, self.headc))
        self.vln = nn.LayerNorm((heads, 1, self.headc))

        self.act = nn.GELU()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Applies Galerkin-type attention.

        Args:
            input: Input tensor [B, C, H, W].

        Returns:
            Output tensor [B, C, H, W] with attention applied.
        """
        b, c, h, w = input.shape
        qkv = self.qkv_proj(input).permute(0, 2, 3, 1).reshape(b, h * w, self.heads, 3 * self.headc)
        q, k, v = qkv.permute(0, 2, 1, 3).chunk(3, dim=-1)  # [B, heads, h*w, headc]

        k = self.kln(k)  # [B, heads, h*w, headc]
        v = self.vln(v)  # [B, heads, h*w, headc]

        v = torch.matmul(k.transpose(-2, -1), v) / (h * w)  # [B, heads, headc, headc]
        v = torch.matmul(q, v).permute(0, 2, 1, 3).reshape(b, h, w, c)  # [B, h, w, C]

        ret = v.permute(0, 3, 1, 2) + input  # [B, C, h, w]
        return self.o_proj2(self.act(self.o_proj1(ret))) + input

# endregion


# region Pixel Attention Module

class PixelAttentionModule(nn.Module):
    """Pixel Attention Module for spatial feature enhancement.

    Args:
        channels: Number of input channels.
        reduction_ratio: Channel reduction ratio.
        kernel_size: Size of the convolution kernel.
    """

    def __init__(
        self,
        channels       : int,
        reduction_ratio: int,
        kernel_size    : _size_2_t
    ):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(
                in_channels  = channels,
                out_channels = channels // reduction_ratio,
                kernel_size  = kernel_size
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels  = channels // reduction_ratio,
                out_channels = 1,
                kernel_size  = kernel_size
            )
        )
        self.act = nn.Sigmoid()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Applies pixel attention to the input.

        Args:
            input: Input tensor [B, C, H, W].

        Returns:
            Output tensor [B, C, H, W] with pixel attention applied.
        """
        return input * self.act(self.fc(input))


PAM = PixelAttentionModule

# endregion


# region Squeeze Excitation

class SqueezeExciteC(nn.Module):
    """Squeeze and Excite layer using Conv2d from 'Squeeze and Excitation' paper.

    Args:
        channels: Number of input channels.
        reduction_ratio: Channel reduction ratio. Default is ``16``.
        bias: If ``True``, adds bias to convolutions. Default is ``False``.
        
    References:
        https://amaarora.github.io/2020/07/24/SeNet.html#squeeze-and-excitation-block-in-pytorch
        https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py
    """
    
    def __init__(
        self,
        channels       : int,
        reduction_ratio: int  = 16,
        bias           : bool = False,
    ):
        super().__init__()
        self.avg_pool   = nn.AdaptiveAvgPool2d(1)  # squeeze
        self.excitation = nn.Sequential(
            nn.Conv2d(
                in_channels  = channels,
                out_channels = channels  // reduction_ratio,
                kernel_size  = 1,
                bias         = bias,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels  = channels  // reduction_ratio,
                out_channels = channels,
                kernel_size  = 1,
                bias         = bias,
            ),
            nn.Sigmoid()
        )
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Applies squeeze and excite attention.

        Args:
            input: Input tensor [B, C, H, W].

        Returns:
            Output tensor [B, C, H, W] with channel attention applied.
        """
        return input * self.excitation(self.avg_pool(input))


class SqueezeExciteL(nn.Module):
    """Squeeze and Excite layer using Linear from 'Squeeze and Excitation' paper.

    Args:
        channels: Number of input channels.
        reduction_ratio: Channel reduction ratio. Default is ``16``.
        bias: If ``True``, adds bias to linear layers. Default is ``False``.

    References:
        https://amaarora.github.io/2020/07/24/SeNet.html#squeeze-and-excitation-block-in-pytorch
        https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py
    """
    
    def __init__(
        self,
        channels       : int,
        reduction_ratio: int  = 16,
        bias           : bool = False,
    ):
        super().__init__()
        self.avg_pool   = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(
                in_features  = channels,
                out_features = channels // reduction_ratio,
                bias         = bias
            ),
            nn.ReLU(inplace=True),
            nn.Linear(
                in_features  = channels // reduction_ratio,
                out_features = channels,
                bias         = bias
            ),
            nn.Sigmoid()
        )
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Applies squeeze and excite attention.

        Args:
            input: Input tensor [B, C, H, W].

        Returns:
            Output tensor [B, C, H, W] with channel attention applied.
        """
        b, c, _, _ = input.shape
        y = self.avg_pool(input).view(b, c)      # [B, C, 1, 1] -> [B, C]
        y = self.excitation(y).view(b, c, 1, 1)  # [B, C] -> [B, C, 1, 1]
        return input * y


ChannelAttention = SqueezeExciteC

# endregion


# region SimAm

class SimAM(nn.Module):
    """SimAM: Simple, Parameter-Free Attention Module from the paper.

    Args:
        e_lambda: Regularization parameter for energy. Default is ``1e-4``.

    References:
        https://github.com/ZjjConan/SimAM
    """

    def __init__(self, e_lambda: float = 1e-4):
        super().__init__()
        self.e_lambda = e_lambda
        self.act      = nn.Sigmoid()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Applies SimAM attention to the input.

        Args:
            input: Input tensor [B, C, H, W].

        Returns:
            Output tensor [B, C, H, W] with attention applied.
        """
        b, c, h, w = input.size()
        n     = w * h - 1
        d     = (input - input.mean(dim=[2, 3], keepdim=True)).pow(2)  # [B, C, H, W]
        v     = d.sum(dim=[2, 3], keepdim=True) / n   # [B, C, 1, 1]
        e_inv = d / (4 * (v + self.e_lambda)) + 0.5   # [B, C, H, W]
        return input * self.act(e_inv)

# endregion
