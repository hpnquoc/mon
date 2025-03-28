#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements channel attention layers."""

from __future__ import annotations

__all__ = [
    "BAM",
    "CBAM",
    "ChannelAttentionModule",
    "ECA",
    "ECA1d",
    "EfficientChannelAttention",
    "EfficientChannelAttention1d",
    "SimplifiedChannelAttention",
]

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.common_types import _size_2_t


# region Channel Attention

class EfficientChannelAttention(nn.Module):
    """Efficient Channel Attention (ECA) module.

    Args:
        channels: Number of input channels as ``int``.
        kernel_size: Kernel size for 1D convolution as ``int`` or ``tuple[int, int]``.
            Default is ``3``.
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
            input: Input tensor as ``torch.Tensor`` with shape [B, C, H, W].

        Returns:
            Output tensor as ``torch.Tensor`` with shape [B, C, H, W] with
            channel attention applied.
        """
        x = input
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

    def flops(self) -> float:
        """Calculates FLOPs for the module.

        Returns:
            Number of floating-point operations as ``float``.
        """
        return self.channel * self.channel * self.k_size


class EfficientChannelAttention1d(nn.Module):
    """Efficient Channel Attention (ECA) module for 1D inputs.

    Args:
        channels: Number of input channels as ``int``.
        kernel_size: Kernel size for 1D convolution as ``int`` or ``tuple[int, int]``.
            Default is ``3``.
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
            input: Input tensor as ``torch.Tensor`` with shape [B, C, L].

        Returns:
            Output tensor as ``torch.Tensor`` with shape [B, C, L] with
            channel attention applied.
        """
        x = input
        y = self.avg_pool(x)                   # [B, C, 1]
        y = self.conv(y.transpose(-1, -2))     # [B, 1, C] -> [B, 1, C]
        y = self.sigmoid(y.transpose(-1, -2))  # [B, C, 1]
        return x * y.expand_as(x)

    def flops(self) -> float:
        """Calculates FLOPs for the module.

        Returns:
            Number of floating-point operations as ``float``
        """
        return self.channel * self.channel * self.k_size


class SimplifiedChannelAttention(nn.Module):
    """Simplified channel attention from 'Simple Baselines for Image Restoration'.

    Args:
        channels: Number of input/output channels as ``int``.
        bias: Adds bias to convolution if ``True``. Default is ``True``.
        device: Device for the module as ``Any``. Default is ``None``.
        dtype: Data type for the module as ``Any``. Default is ``None``.

    References:
        - https://arxiv.org/pdf/2204.04676.pdf
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
            input: Input tensor as ``torch.Tensor`` with shape [B, C, H, W].

        Returns:
            Output tensor as ``torch.Tensor`` with shape [B, C, H, W] with
            channel attention applied.
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

    Args:
        channels: Number of input channels as ``int``.
        reduction_ratio: Channel reduction ratio as ``int``. Default is ``16``.
        num_layers: Number of hidden layers in channel attention as ``int``.
            Default is ``1``.

    References:
        - https://github.com/Jongchan/attention-module/blob/master/MODELS/bam.py
    """

    class Flatten(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            """Flattens input tensor to [B, -1].

            Args:
                input: Input tensor as ``torch.Tensor`` of any shape.

            Returns:
                Flattened tensor as ``torch.Tensor`` with shape [B, -1].
            """
            return input.view(input.size(0), -1)

    class ChannelAttention(nn.Module):
        """Channel attention submodule for BAM.

        Args:
            channels: Number of input channels as ``int``.
            reduction_ratio: Channel reduction ratio as ``int``. Default is ``16``.
            num_layers: Number of hidden layers as ``int``. Default is ``1``.

        Attributes:
            c_gate: Sequential layer for channel gating as ``nn.Sequential``.
        """
        def __init__(
            self,
            channels       : int,
            reduction_ratio: int = 16,
            num_layers     : int = 1
        ):
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
                input: Input tensor as ``torch.Tensor`` with shape [B, C, H, W].

            Returns:
                Attention weights tensor as ``torch.Tensor`` with shape [B, C, H, W].
            """
            y = F.avg_pool2d(input, input.size()[2:], stride=input.size()[2:])  # [B, C, 1, 1]
            y = self.c_gate(y)  # [B, C]
            return y.unsqueeze(2).unsqueeze(3).expand_as(input)  # [B, C, H, W]

    class SpatialAttention(nn.Module):
        """Spatial attention submodule for BAM.

        Args:
            channels: Number of input channels as ``int``.
            reduction_ratio: Channel reduction ratio as ``int``. Default is ``16``.
            dilation_conv_num: Number of dilated convolutions as ``int``. Default is ``2``.
            dilation_val: Dilation value for convolutions as ``int``. Default is ``4``.

        Attributes:
            s_gate: Sequential layer for spatial gating as ``nn.Sequential``.
        """
        def __init__(
            self,
            channels         : int,
            reduction_ratio  : int = 16,
            dilation_conv_num: int = 2,
            dilation_val     : int = 4
        ):
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
                input: Input tensor as ``torch.Tensor`` with shape [B, C, H, W].

            Returns:
                Attention weights tensor as ``torch.Tensor`` with shape [B, C, H, W].
            """
            return self.s_gate(input).expand_as(input)

    def __init__(
        self,
        channels       : int,
        reduction_ratio: int = 16,
        num_layers     : int = 1
    ):
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
            input: Input tensor as ``torch.Tensor`` with shape [B, C, H, W].

        Returns:
            Output tensor as ``torch.Tensor`` with shape [B, C, H, W] with
            attention applied.
        """
        y = 1 + self.sigmoid(self.channel_att(input) * self.spatial_att(input))
        return input * y


class CBAM(nn.Module):
    """Convolutional Block Attention Module from CBAM paper.

    Args:
        channels: Number of input channels as ``int``.
        reduction_ratio: Channel reduction ratio as ``int``. Default is ``16``.
        pool_types: Pooling layer types as ``list[str]``. Default is ``["avg", "max"]``.
        spatial: Includes spatial attention if ``True``. Default is ``True``.

    References:
        - https://github.com/Jongchan/attention-module/blob/master/MODELS/cbam.py
    """

    class Flatten(nn.Module):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            """Flattens input tensor to [B, -1].

            Args:
                input: Input tensor as ``torch.Tensor`` of any shape.

            Returns:
                Flattened tensor as ``torch.Tensor`` with shape [B, -1].
            """
            return input.view(input.size(0), -1)

    class ChannelAttention(nn.Module):
        """Channel attention submodule for CBAM.

        Args:
            channels: Number of input channels as ``int``.
            reduction_ratio: Channel reduction ratio as ``int``. Default is ``16``.
            pool_types: Pooling layer types as ``list[str]``. Default is ``["avg", "max"]``.
        """
        def __init__(
            self,
            channels       : int,
            reduction_ratio: int = 16,
            pool_types     : list[str] = ["avg", "max"]
        ):
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
                input: Input tensor as ``torch.Tensor`` with shape [B, C, H, W].

            Returns:
                Output tensor as ``torch.Tensor`` with shape [B, C, H, W] with
                channel attention applied.
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
        """Spatial attention submodule for CBAM.

        Attributes:
            spatial: Sequential layer for spatial gating as ``nn.Sequential``.
        """
        def __init__(self):
            super().__init__()
            self.spatial = nn.Sequential(
                nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
                nn.BatchNorm2d(1),
                nn.Sigmoid()
            )

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            """Applies spatial attention.

            Args:
                input: Input tensor as ``torch.Tensor`` with shape [B, C, H, W].

            Returns:
                Output tensor as ``torch.Tensor`` with shape [B, C, H, W] with
                spatial attention applied.
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
            input: Input tensor as ``torch.Tensor`` with shape [B, C, H, W].

        Returns:
            Output tensor as ``torch.Tensor`` with shape [B, C, H, W] with
            attention applied.
        """
        return self.spatial_att(self.channel_att(input))


class ChannelAttentionModule(nn.Module):
    """Channel Attention Module for feature enhancement.

    Args:
        channels: Number of input channels as ``int``.
        reduction_ratio: Channel reduction ratio as ``int``.
        stride: Stride of the first convolution as ``int``. Default is ``1``.
        padding: Padding of the first convolution as ``int``. Default is ``0``.
        dilation: Dilation of the convolutions as ``int``. Default is ``1``.
        groups: Number of groups in the convolutions as ``int``. Default is ``1``.
        bias: Adds bias to convolutions if ``True``. Default is ``True``.
        padding_mode: Padding mode for convolutions as ``str``. Default is ``"zeros"``.
        device: Device for the module as ``Any``. Default is ``None``.
        dtype: Data type for the module as ``Any``. Default is ``None``.
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
            input: Input tensor as ``torch.Tensor`` with shape [B, C, H, W].

        Returns:
            Output tensor as ``torch.Tensor`` with shape [B, C, H, W] with
            channel attention applied.
        """
        return input * self.excitation(self.avg_pool(input))

# endregion
