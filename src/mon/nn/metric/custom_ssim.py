#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Custom SSIM for PyTorch.

This module implements the custom SSIM which is used as the loss function in the
paper: "ESDNet: Efficient and Scalable Deep Net for Small Object Detection".

References:
    https://github.com/MingTian99/ESDNet/blob/master/utils/image_utils.py
"""

from __future__ import annotations

__all__ = [
    "SSIM",
]

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable


def _gaussian(window_size: int, sigma: float) -> torch.Tensor:
    gauss = torch.Tensor([
        math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
        for x in range(window_size)
    ])
    return gauss / gauss.sum()


def _create_window(window_size: int, channel: int, sigma: float = 1.5) -> torch.Tensor:
    _1d_window = _gaussian(window_size, sigma).unsqueeze(1)
    _2d_window = _1d_window.mm(_1d_window.t()).float().unsqueeze(0).unsqueeze(0)
    window     = Variable(_2d_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(
    image1      : torch.Tensor,
    image2      : torch.Tensor,
    window      : torch.Tensor,
    window_size : int  = 11,
    channel     : int  = 1,
    k           : tuple[float, float] = (0.01, 0.03),
    size_average: bool = True,
) -> torch.Tensor:
    mu1 = F.conv2d(image1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(image2, window, padding=window_size // 2, groups=channel)
    
    mu1_sq  = mu1.pow(2)
    mu2_sq  = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(image1 * image1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(image2 * image2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12   = F.conv2d(image1 * image2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = k[0] ** 2
    C2 = k[1] ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(nn.Module):

    def __init__(
        self,
        window_size : int  = 11,
        channel     : int  = 1,
        k           : tuple[float, float] = (0.01, 0.03),
        size_average: bool = True,
    ):
        super().__init__()
        self.window_size  = window_size
        self.size_average = size_average
        self.channel      = channel
        self.k            = k
        self.window       = _create_window(window_size, self.channel)
    
    def forward(self, image1: torch.Tensor, image2: torch.Tensor) -> torch.Tensor:
        _, channel, _, _ = image1.size()
        if channel == self.channel and self.window.data.type() == image1.data.type():
            window = self.window
        else:
            window = _create_window(self.window_size, channel)

            if image1.is_cuda:
                window = window.cuda(image1.get_device())
            window = window.type_as(image1)

            self.window  = window
            self.channel = channel

        return _ssim(
            image1       = image1,
            image2       = image2,
            window       = window,
            window_size  = self.window_size,
            channel      = channel,
            k            = self.k,
            size_average = self.size_average
        )
