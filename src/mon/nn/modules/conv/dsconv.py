#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Depthwise Separable Convolution Module.

This module implements depthwise separable convolutional layers.
"""

from __future__ import annotations

__all__ = [
    "DSConv2d",
    "DSConv2dReLU",
    "DSConvAct2d",
    "DWConv2d",
    "DepthwiseConv2d",
    "DepthwiseSeparableConv2d",
    "DepthwiseSeparableConv2dReLU",
    "DepthwiseSeparableConvAct2d",
    "PWConv2d",
    "PointwiseConv2d",
]

from typing import Any

import torch
from torch import nn
from torch.nn.common_types import _size_2_t


# region Depthwise Separable Convolution

class DepthwiseConv2d(nn.Module):
    """Applies depthwise 2D convolution.

    Args:
        in_channels: Number of input channels.
        kernel_size: Size of the convolution kernel.
        stride: Stride of the convolution. Default is ``1``.
        padding: Padding size or mode. Default is ``0``.
        dilation: Dilation of the convolution. Default is ``1``.
        bias: If ``True``, adds bias to convolution. Default is ``True``.
        padding_mode: Padding mode for convolution. Default is ``"zeros"``.
        device: Device for the module. Default is ``None``.
        dtype: Data type for the module. Default is ``None``.
    """

    def __init__(
        self,
        in_channels : int,
        kernel_size : _size_2_t,
        stride      : _size_2_t = 1,
        padding     : _size_2_t | str = 0,
        dilation    : _size_2_t = 1,
        bias        : bool = True,
        padding_mode: str  = "zeros",
        device      : Any  = None,
        dtype       : Any  = None
    ):
        super().__init__()
        self.dw_conv = nn.Conv2d(
            in_channels  = in_channels,
            out_channels = in_channels,
            kernel_size  = kernel_size,
            stride       = stride,
            padding      = padding,
            dilation     = dilation,
            groups       = in_channels,
            bias         = bias,
            padding_mode = padding_mode,
            device       = device,
            dtype        = dtype
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Applies depthwise convolution.

        Args:
            input: Input tensor ``[B, C_in, H, W]``.

        Returns:
            Output tensor ``[B, C_in, H_out, W_out]``.
        """
        return self.dw_conv(input)


class PointwiseConv2d(nn.Module):
    """Applies pointwise 2D convolution (1x1 kernel).

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        stride: Stride of the convolution. Default is ``1``.
        padding: Padding size or mode. Default is ``0``.
        dilation: Dilation of the convolution. Default is ``1``.
        groups: Number of groups in convolution. Default is ``1``.
        bias: If ``True``, adds bias to convolution. Default is ``True``.
        padding_mode: Padding mode for convolution. Default is ``"zeros"``.
        device: Device for the module. Default is ``None``.
        dtype: Data type for the module. Default is ``None``.
    """
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        stride      : _size_2_t = 1,
        padding     : _size_2_t | str = 0,
        dilation    : _size_2_t = 1,
        groups      : int  = 1,
        bias        : bool = True,
        padding_mode: str  = "zeros",
        device      : Any  = None,
        dtype       : Any  = None
    ):
        super().__init__()
        self.pw_conv = nn.Conv2d(
            in_channels  = in_channels,
            out_channels = out_channels,
            kernel_size  = 1,
            stride       = stride,
            padding      = padding,
            dilation     = dilation,
            groups       = groups,
            bias         = bias,
            padding_mode = padding_mode,
            device       = device,
            dtype        = dtype
        )
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Applies pointwise convolution.

        Args:
            input: Input tensor ``[B, C_in, H, W]``.

        Returns:
            Output tensor ``[B, C_out, H_out, W_out]``.
        """
        return self.pw_conv(input)
    

class DepthwiseSeparableConv2d(nn.Module):
    """Applies depthwise separable 2D convolution.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Size of the depthwise kernel.
        stride: Stride of the convolution. Default is ``1``.
        padding: Padding size or mode. Default is ``0``.
        dilation: Dilation of the convolution. Default is ``1``.
        bias: If ``True``, adds bias to convolutions. Default is ``True``.
        padding_mode: Padding mode for convolutions. Default is ``"zeros"``.
        device: Device for the module. Default is ``None``.
        dtype: Data type for the module. Default is ``None``.
    """

    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : _size_2_t,
        stride      : _size_2_t = 1,
        padding     : _size_2_t | str = 0,
        dilation    : _size_2_t = 1,
        bias        : bool = True,
        padding_mode: str  = "zeros",
        device      : Any  = None,
        dtype       : Any  = None
    ):
        super().__init__()
        self.dw_conv = nn.Conv2d(
            in_channels  = in_channels,
            out_channels = in_channels,
            kernel_size  = kernel_size,
            stride       = stride,
            padding      = padding,
            dilation     = dilation,
            groups       = in_channels,
            bias         = bias,
            padding_mode = padding_mode,
            device       = device,
            dtype        = dtype
        )
        self.pw_conv = nn.Conv2d(
            in_channels  = in_channels,
            out_channels = out_channels,
            kernel_size  = 1,
            bias         = bias,
            padding_mode = padding_mode,
            device       = device,
            dtype        = dtype
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Applies depthwise then pointwise convolution.

        Args:
            input: Input tensor ``[B, C_in, H, W]``.

        Returns:
            Output tensor ``[B, C_out, H_out, W_out]``.
        """
        y = self.dw_conv(input)
        y = self.pw_conv(y)
        return y


class DepthwiseSeparableConvAct2d(nn.Module):
    """Applies depthwise separable 2D convolution with activation.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Size of the depthwise kernel.
        stride: Stride of the convolution. Default is ``1``.
        padding: Padding size or mode. Default is ``0``.
        dilation: Dilation of the convolution. Default is ``1``.
        bias: If ``True``, adds bias to convolutions. Default is ``True``.
        padding_mode: Padding mode for convolutions. Default is ``"zeros"``.
        device: Device for the module. Default is ``None``.
        dtype: Data type for the module. Default is ``None``.
        act_layer: Activation layer class. Default is ``nn.ReLU``.
    """

    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : _size_2_t,
        stride      : _size_2_t = 1,
        padding     : _size_2_t | str = 0,
        dilation    : _size_2_t = 1,
        bias        : bool      = True,
        padding_mode: str       = "zeros",
        device      : Any       = None,
        dtype       : Any       = None,
        act_layer   : nn.Module = nn.ReLU
    ):
        super().__init__()
        self.ds_conv = DepthwiseSeparableConv2d(
            in_channels  = in_channels,
            out_channels = out_channels,
            kernel_size  = kernel_size,
            stride       = stride,
            padding      = padding,
            dilation     = dilation,
            bias         = bias,
            padding_mode = padding_mode,
            device       = device,
            dtype        = dtype
        )
        self.act = act_layer()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Applies depthwise separable convolution and activation.

        Args:
            input: Input tensor ``[B, C_in, H, W]``.

        Returns:
            Output tensor ``[B, C_out, H_out, W_out]`` after activation.
        """
        y = self.ds_conv(input)
        y = self.act(y)
        return y


class DepthwiseSeparableConv2dReLU(nn.Module):
    """Applies depthwise separable 2D convolution with ReLU activation.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Size of the depthwise kernel.
        stride: Stride of the convolution. Default is ``1``.
        padding: Padding size or mode. Default is ``0``.
        dilation: Dilation of the convolution. Default is ``1``.
        bias: If ``True``, adds bias to convolutions. Default is ``True``.
        padding_mode: Padding mode for convolutions. Default is ``"zeros"``.
        device: Device for the module. Default is ``None``.
        dtype: Data type for the module. Default is ``None``.
    """

    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : _size_2_t,
        stride      : _size_2_t = 1,
        padding     : _size_2_t | str = 0,
        dilation    : _size_2_t = 1,
        bias        : bool      = True,
        padding_mode: str       = "zeros",
        device      : Any       = None,
        dtype       : Any       = None
    ):
        super().__init__()
        self.ds_conv = DepthwiseSeparableConv2d(
            in_channels  = in_channels,
            out_channels = out_channels,
            kernel_size  = kernel_size,
            stride       = stride,
            padding      = padding,
            dilation     = dilation,
            bias         = bias,
            padding_mode = padding_mode,
            device       = device,
            dtype        = dtype
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Applies depthwise separable convolution and ReLU.

        Args:
            input: Input tensor ``[B, C_in, H, W]``.

        Returns:
            Output tensor ``[B, C_out, H_out, W_out]`` after ReLU.
        """
        y = self.ds_conv(input)
        y = self.act(y)
        return y


DWConv2d     = DepthwiseConv2d
PWConv2d     = PointwiseConv2d
DSConv2d     = DepthwiseSeparableConv2d
DSConvAct2d  = DepthwiseSeparableConvAct2d
DSConv2dReLU = DepthwiseSeparableConv2dReLU

# endregion
