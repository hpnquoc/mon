#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements Fast Fourier Convolution layers."""

from __future__ import annotations

__all__ = [
    "FFConv2d",
    "FFConv2dNormAct",
    "FFConv2dSE",
    "FastFourierConv2d",
    "FastFourierConv2dNormAct",
    "FastFourierConv2dSE",
    "FourierUnit",
    "FourierUnit2d",
    "FourierUnit3d",
    "SpectralTransform2d",
]

from typing import Any

import torch
from torch import nn
from torch.nn.common_types import _size_2_t


# region Fourier Transform

class FourierUnit(nn.Module):
    """Fourier transform unit from Fast Fourier Convolution paper.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        groups: Number of groups in convolution. Default is ``1``.
        ffc3d: If ``True``, uses 3D FFT (called by FourierUnit3d). Default is ``False``.
        fft_norm: FFT normalization mode (``'forward'``, ``'backward'``, ``'ortho'``).
            Default is ``'ortho'``.

    References:
        - https://github.com/pkumivision/FFC
    """

    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        groups      : int  = 1,
        ffc3d       : bool = False,
        fft_norm    : str  = "ortho"
    ):
        super().__init__()
        self.groups   = groups
        self.ffc3d    = ffc3d
        self.fft_norm = fft_norm
        self.conv     = nn.Conv2d(
            in_channels  = in_channels * 2,
            out_channels = out_channels * 2,
            kernel_size  = 1,
            stride       = 1,
            padding      = 0,
            groups       = self.groups,
            bias         = False
        )
        self.bn   = nn.BatchNorm2d(out_channels * 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Applies Fourier transform, convolution, and inverse transform.

        Args:
            input: Input tensor [B, C, H, W] or ``[B, C, D, H, W]``.

        Returns:
            Output tensor with same shape as input.
        """
        x          = input
        b, c, h, w = x.size()
        fft_dim    = (-3, -2, -1) if self.ffc3d else (-2, -1)

        ffted = torch.fft.rfftn(x, dim=fft_dim, norm=self.fft_norm)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()  # [b, c, 2, h, w/2+1]
        ffted = ffted.view((b, -1,) + ffted.size()[3:])

        ffted = self.conv(ffted)  # [b, c*2, h, w/2+1]
        ffted = self.relu(self.bn(ffted))

        ffted = ffted.view((b, -1, 2,) + ffted.size()[2:]).permute(0, 1, 3, 4, 2).contiguous()  # [b, c, h, w/2+1, 2]
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])

        ifft_shape_slice = x.shape[-3:] if self.ffc3d else x.shape[-2:]
        y = torch.fft.irfftn(ffted, s=ifft_shape_slice, dim=fft_dim, norm=self.fft_norm)
        return y


class FourierUnit2d(FourierUnit):
    """2D Fourier transform unit from Fast Fourier Convolution.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        groups: Number of groups in convolution. Default is ``1``.
        fft_norm: FFT normalization mode (``'forward'``, ``'backward'``, ``'ortho'``).
            Default is ``'ortho'``.
    """

    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        groups      : int  = 1,
        fft_norm    : str  = "ortho"
    ):
        super().__init__(
            in_channels  = in_channels,
            out_channels = out_channels,
            groups       = groups,
            ffc3d        = False,
            fft_norm     = fft_norm
        )


class FourierUnit3d(FourierUnit):
    """3D Fourier transform unit from Fast Fourier Convolution.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        groups: Number of groups in convolution. Default is ``1``.
        fft_norm: FFT normalization mode (``'forward'``, ``'backward'``, ``'ortho'``).
            Default is ``'ortho'``.
    """

    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        groups      : int  = 1,
        fft_norm    : str  = "ortho"
    ):
        super().__init__(
            in_channels  = in_channels,
            out_channels = out_channels,
            groups       = groups,
            ffc3d        = True,
            fft_norm     = fft_norm
        )
        

class SpectralTransform2d(nn.Module):
    """Spectral transform unit from Fast Fourier Convolution paper.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        stride: Stride of the convolution. Default is ``1``.
        groups: Number of groups in convolution. Default is ``1``.
        enable_lfu: If ``True``, includes local Fourier unit. Default is ``True``.
        fft_norm: FFT normalization mode (``'forward'``, ``'backward'``, ``'ortho'``).
            Default is ``'ortho'``.

    References:
        - https://github.com/pkumivision/FFC
    """

    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        stride      : _size_2_t = 1,
        groups      : int       = 1,
        enable_lfu  : bool      = True,
        fft_norm    : str       = "ortho"
    ):
        super().__init__()
        self.enable_lfu = enable_lfu
        self.stride     = stride
        self.downsample = (
            nn.AvgPool2d(kernel_size=(2, 2), stride=2)
            if self.stride == 2 else nn.Identity()
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels  = in_channels,
                out_channels = out_channels // 2,
                kernel_size  = 1,
                groups       = groups,
                bias         = False
            ),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True)
        )
        self.fu = FourierUnit2d(
            in_channels  = out_channels // 2,
            out_channels = out_channels // 2,
            groups       = groups,
            fft_norm     = fft_norm
        )
        if self.enable_lfu:
            self.lfu = FourierUnit2d(
                in_channels  = out_channels // 2,
                out_channels = out_channels // 2,
                groups       = groups,
                fft_norm     = fft_norm
            )
        self.conv2 = nn.Conv2d(
            in_channels  = out_channels // 2,
            out_channels = out_channels,
            kernel_size  = 1,
            groups       = groups,
            bias         = False
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Applies spectral transform with Fourier units.

        Args:
            input: Input tensor [B, C_in, H, W].

        Returns:
            Output tensor [B, C_out, H_out, W_out].
        """
        x = input
        x = self.downsample(x)
        x = self.conv1(x)
        y = self.fu(x)
        if self.enable_lfu:
            b, c, h, w = x.shape
            split_no   = 2
            split_s_h  = h // split_no
            split_s_w  = w // split_no
            xs         = torch.cat(torch.split(x[:, :c // 4], split_s_h, dim=-2), dim=1).contiguous()
            xs         = torch.cat(torch.split(xs, split_s_w, dim=-1), dim=1).contiguous()
            xs         = self.lfu(xs)
            xs         = xs.repeat(1, 1, split_no, split_no).contiguous()
        else:
            xs = 0
        y = self.conv2(x + y + xs)
        return y

# endregion


# region Fast-Fourier Convolution

class FastFourierConv2d(nn.Module):
    """Fast Fourier convolution from FFC paper.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Size of the convolution kernel.
        ratio_g_in: Ratio of global input channels. Range [0, 1].
        ratio_g_out: Ratio of global output channels. Range [0, 1].
        stride: Stride of the convolution. Default is ``1``.
        padding: Padding size for convolutions. Default is ``0``.
        dilation: Dilation of the convolution. Default is ``1``.
        groups: Number of groups in convolution. Default is ``1``.
        bias: If ``True``, adds bias to convolutions. Default is ``False``.
        padding_mode: Padding mode for convolutions. Default is ``"zeros"``.
        enable_lfu: If ``True``, enables local Fourier unit. Default is ``True``.
        fft_norm: FFT normalization mode (``'forward'``, ``'backward'``, ``'ortho'``).
            Default is ``'ortho'``.
    
    References:
        - https://github.com/pkumivision/FFC
    """

    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : _size_2_t,
        ratio_g_in  : float,
        ratio_g_out : float,
        stride      : _size_2_t = 1,
        padding     : _size_2_t = 0,
        dilation    : _size_2_t = 1,
        groups      : int       = 1,
        bias        : bool      = False,
        padding_mode: str       = "zeros",
        enable_lfu  : bool      = True,
        fft_norm    : str       = "ortho"
    ):
        super().__init__()
        if stride not in [1, 2]:
            raise ValueError(f"[stride] must be 1 or 2, got [{stride}]")
        self.stride      = stride
        self.ratio_g_in  = ratio_g_in
        self.ratio_g_out = ratio_g_out

        in_c_g  = int(in_channels * self.ratio_g_in)
        in_c_l  = in_channels - in_c_g
        out_c_g = int(out_channels * self.ratio_g_out)
        out_c_l = out_channels - out_c_g

        self.conv_l2l = (
            nn.Conv2d(
                in_channels  = in_c_l,
                out_channels = out_c_l,
                kernel_size  = kernel_size,
                stride       = stride,
                padding      = padding,
                dilation     = dilation,
                groups       = groups,
                bias         = bias,
                padding_mode = padding_mode
            ) if in_c_l > 0 and out_c_l > 0 else nn.Identity()
        )
        self.conv_l2g = (
            nn.Conv2d(
                in_channels  = in_c_l,
                out_channels = out_c_g,
                kernel_size  = kernel_size,
                stride       = stride,
                padding      = padding,
                dilation     = dilation,
                groups       = groups,
                bias         = bias,
                padding_mode = padding_mode
            ) if in_c_l > 0 and out_c_g > 0 else nn.Identity()
        )
        self.conv_g2l = (
            nn.Conv2d(
                in_channels  = in_c_g,
                out_channels = out_c_l,
                kernel_size  = kernel_size,
                stride       = stride,
                padding      = padding,
                dilation     = dilation,
                groups       = groups,
                bias         = bias,
                padding_mode = padding_mode
            ) if in_c_g > 0 and out_c_l > 0 else nn.Identity()
        )
        self.conv_g2g = (
            SpectralTransform2d(
                in_channels  = in_c_g,
                out_channels = out_c_g,
                stride       = stride,
                groups       = 1 if groups == 1 else groups // 2,
                enable_lfu   = enable_lfu,
                fft_norm     = fft_norm
            ) if in_c_g > 0 and out_c_g > 0 else nn.Identity()
        )

    def forward(self, input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Applies fast Fourier convolution with local and global paths.

        Args:
            input: Input tensor [B, C_in, H, W] or tuple of local/global tensors.

        Returns:
            Tuple of (local output ``[B, C_out_l, H_out, W_out]``,
            global output ``[B, C_out_g, H_out, W_out]``).
        """
        x = input
        x_l, x_g = x if isinstance(x, (tuple, list)) else (x, 0)
        y_l = self.conv_l2l(x_l) + self.conv_g2l(x_g) if self.ratio_g_out != 1 else 0
        y_g = self.conv_l2g(x_l) + self.conv_g2g(x_g) if self.ratio_g_out != 0 else 0
        return y_l, y_g


class FastFourierConv2dNormAct(nn.Module):
    """Fast Fourier convolution with normalization and activation from FFC paper.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Size of the convolution kernel.
        ratio_g_in: Ratio of global input channels. Range [0, 1].
        ratio_g_out: Ratio of global output channels. Range [0, 1].
        stride: Stride of the convolution. Default is ``1``.
        padding: Padding size for convolutions. Default is ``0``.
        dilation: Dilation of the convolution. Default is ``1``.
        groups: Number of groups in convolution. Default is ``1``.
        bias: If True, adds bias to convolutions. Default is ``False``.
        padding_mode: Padding mode for convolutions. Default is ``"zeros"``.
        norm_layer: Normalization layer class. Default is ``nn.BatchNorm2d``.
        act_layer: Activation layer class. Default is ``nn.Identity``.
        enable_lfu: If True, enables local Fourier unit. Default is ``True``.
        fft_norm: FFT normalization mode (``'forward'``, ``'backward'``, ``'ortho'``).
            Default is ``'ortho'``.

    Notes:
        Mimics torchvision.ops.misc.Conv2dNormActivation naming.
    
    References:
        - https://github.com/pkumivision/FFC
    """

    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : _size_2_t,
        ratio_g_in  : float,
        ratio_g_out : float,
        stride      : _size_2_t = 1,
        padding     : _size_2_t = 0,
        dilation    : _size_2_t = 1,
        groups      : int       = 1,
        bias        : bool      = False,
        padding_mode: str       = "zeros",
        norm_layer  : Any       = nn.BatchNorm2d,
        act_layer   : Any       = nn.Identity,
        enable_lfu  : bool      = True,
        fft_norm    : str       = "ortho"
    ):
        super().__init__()
        self.ffc = FastFourierConv2d(
            in_channels  = in_channels,
            out_channels = out_channels,
            kernel_size  = kernel_size,
            ratio_g_in   = ratio_g_in,
            ratio_g_out  = ratio_g_out,
            stride       = stride,
            padding      = padding,
            dilation     = dilation,
            groups       = groups,
            bias         = bias,
            padding_mode = padding_mode,
            enable_lfu   = enable_lfu,
            fft_norm     = fft_norm
        )
        self.norm_l = (
            nn.Identity() if ratio_g_out == 1 else norm_layer(int(out_channels * (1 - ratio_g_out)))
        )
        self.norm_g = (
            nn.Identity() if ratio_g_out == 0 else norm_layer(int(out_channels * ratio_g_out))
        )
        self.act_l = (
            nn.Identity() if ratio_g_out == 1 else act_layer(inplace=True)
        )
        self.act_g = (
            nn.Identity() if ratio_g_out == 0 else act_layer(inplace=True)
        )

    def forward(self, input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Applies FFC, normalization, and activation.

        Args:
            input: Input tensor [B, C_in, H, W] or tuple of local/global tensors.

        Returns:
            Tuple of (local output ``[B, C_out_l, H_out, W_out]``,
            global output ``[B, C_out_g, H_out, W_out]``).
        """
        y_l, y_g = self.ffc(input)
        y_l      = self.act_l(self.norm_l(y_l))
        y_g      = self.act_g(self.norm_g(y_g))
        return y_l, y_g


class FastFourierConv2dSE(nn.Module):
    """Squeeze and Excitation block for Fast Fourier Convolution from FFC paper.

    Args:
        channels: Total number of channels.
        ratio_g: Ratio of global channels. Range [0, 1].

    References:
        - https://github.com/pkumivision/FFC
    """

    def __init__(
        self,
        channels: int,
        ratio_g : float
    ):
        """Initializes the FastFourierConv2dSE module.

        Args:
            channels: Total number of channels.
            ratio_g: Ratio of global channels. Range [0, 1].
        """
        super().__init__()
        in_cg = int(channels * ratio_g)
        in_cl = channels - in_cg
        r     = 16

        self.avgpool  = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1    = nn.Conv2d(
            in_channels  = channels,
            out_channels = channels // r,
            kernel_size  = 1,
            bias         = True
        )
        self.relu1    = nn.ReLU(inplace=True)
        self.conv_a2l = (
            nn.Conv2d(
                in_channels  = channels // r,
                out_channels = in_cl,
                kernel_size  = 1,
                bias         = True
            ) if in_cl > 0 else None
        )
        self.conv_a2g = (
            nn.Conv2d(
                in_channels  = channels // r,
                out_channels = in_cg,
                kernel_size  = 1,
                bias         = True
            ) if in_cg > 0 else None
        )
        self.sigmoid  = nn.Sigmoid()
    
    def forward(self, input: torch.Tensor | tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Applies squeeze and excitation to local and global paths.

        Args:
            input: Input tensor [B, C, H, W] or tuple of
                (local ``[B, C_l, H, W]``, global ``[B, C_g, H, W]``).

        Returns:
            Tuple of (local output ``[B, C_l, H_out, W_out]``,
            global output ``[B, C_g, H_out, W_out]``).
        """
        x = input
        x_l, x_g = x if isinstance(x, (tuple, list)) else (x, 0)
        x   = x_l if isinstance(x_g, int) else torch.cat([x_l, x_g], dim=1)
        x   = self.avgpool(x)
        x   = self.relu1(self.conv1(x))
        y_l = 0 if self.conv_a2l is None else x_l * self.sigmoid(self.conv_a2l(x))
        y_g = 0 if self.conv_a2g is None else x_g * self.sigmoid(self.conv_a2g(x))
        return y_l, y_g


FFConv2d        = FastFourierConv2d
FFConv2dNormAct = FastFourierConv2dNormAct
FFConv2dSE      = FastFourierConv2dSE

# endregion
