#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements guided filter."""

__all__ = [
    "ConvGuidedFilter",
    "DeepConvGuidedFilter",
    "DeepGuidedFilter",
    "FastGuidedFilter",
    "GuidedFilter",
    "guided_filter",
]

import numpy as np
import torch
from cv2 import ximgproc
from plum import dispatch
from torch.autograd import Variable

from mon import core, nn
from mon.nn import functional as F, init
from mon.vision import geometry
from mon.vision.filtering.box_filter import BoxFilter


# ----- Guided Filter -----
def guided_filter(
    image : torch.Tensor | np.ndarray,
    guide : torch.Tensor | np.ndarray,
    radius: int,
    eps   : float = 1e-8
) -> torch.Tensor | np.ndarray:
    """Applies guided filter to an image.

    Args:
        image: RGB image as ``torch.Tensor`` [B, C, H, W], range [0.0, 1.0], or
            ``numpy.ndarray`` [H, W, C], range [0, 255].
        guide: Guidance image, same shape as image.
        radius: Radius of filter (kernel_size = radius * 2 + 1, e.g., 1, 2, 4, 8).
        eps: Sharpness control value. Default is ``1e-8``.
    
    Returns:
        Filtered image.
    
    Raises:
        TypeError: If ``image`` and ``guide`` types differ or ``image`` type is invalid.
        AssertionError: If tensor shapes or sizes are incompatible.
    
    References:
        - https://github.com/wuhuikai/DeepGuidedFilter/blob/master/GuidedFilteringLayer/GuidedFilter_PyTorch/guided_filter_pytorch/guided_filter.py
    """
    if type(image) != type(guide):
        raise TypeError(f"[image] and [guide] must have the same type, "
                        f"got {type(image)} and {type(guide)}.")

    if isinstance(image, torch.Tensor):
        x = image
        y = guide
        box_filter = BoxFilter(radius=radius)
        n_x, c_x, h_x, w_x = x.size()
        n_y, c_y, h_y, w_y = y.size()
        assert n_x == n_y
        assert c_x == 1 or c_x == c_y
        assert h_x == h_y and w_x == w_y
        assert h_x > 2 * radius + 1 and w_x > 2 * radius + 1
        N      = box_filter(Variable(x.data.new().resize_((1, 1, h_x, w_x)).fill_(1.0)))
        mean_x = box_filter(x) / N
        mean_y = box_filter(y) / N
        cov_xy = box_filter(x * y) / N - mean_x * mean_y
        var_x  = box_filter(x * x) / N - mean_x * mean_x
        A      = cov_xy / (var_x + eps)
        b      = mean_y - A * mean_x
        mean_A = box_filter(A) / N
        mean_b = box_filter(b) / N
        return mean_A * x + mean_b
    elif isinstance(image, np.ndarray):
        return ximgproc.guidedFilter(guide=guide, src=image, radius=radius, eps=eps)
    else:
        raise TypeError(f"[image] must be torch.Tensor or numpy.ndarray, got {type(image)}.")


class GuidedFilter(nn.Module):
    """Applies guided filtering to an image.

    Args:
        radius: Radius of filter (kernel_size = radius * 2 + 1, e.g., 1, 2, 4, 8).
        eps: Sharpness control value. Default is ``1e-8``.
    
    References:
        - https://github.com/wuhuikai/DeepGuidedFilter/blob/master/GuidedFilteringLayer/GuidedFilter_PyTorch/guided_filter_pytorch/guided_filter.py
    """

    def __init__(self, radius: int, eps: float = 1e-8):
        super().__init__()
        self.radius     = radius
        self.eps        = eps
        self.box_filter = BoxFilter(self.radius)

    def forward(self, image: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        """Filters the image using a guidance image.

        Args:
            image: Image as ``torch.Tensor`` in [B, C, H, W] format.
            guide: Guidance image, same shape as ``image``.
        
        Returns:
            Filtered image.
        
        Raises:
            AssertionError: If tensor shapes or sizes are incompatible.
        """
        x = image
        y = guide
        n_x, c_x, h_x, w_x = x.size()
        n_y, c_y, h_y, w_y = y.size()
        assert n_x == n_y
        assert c_x == 1 or c_x == c_y
        assert h_x == h_y and w_x == w_y
        assert h_x > 2 * self.radius + 1 and w_x > 2 * self.radius + 1
        N      = self.box_filter(torch.ones(1, 1, h_x, w_x, device=x.device))
        mean_x = self.box_filter(x) / N
        mean_y = self.box_filter(y) / N
        cov_xy = self.box_filter(x * y) / N - mean_x * mean_y
        var_x  = self.box_filter(x * x) / N - mean_x * mean_x
        A      = cov_xy / (var_x + self.eps)
        b      = mean_y - A * mean_x
        mean_A = self.box_filter(A) / N
        mean_b = self.box_filter(b) / N
        return mean_A * x + mean_b


class FastGuidedFilter(nn.Module):
    """Applies fast guided filtering to an image.

    Args:
        radius: Radius of filter (kernel_size = radius * 2 + 1, e.g., 1, 2, 4, 8).
        eps: Sharpness control value. Default is ``1e-8``.
        downscale: Downscale factor for low-resolution input. Default is ``8``.
    
    References:
        - https://github.com/wuhuikai/DeepGuidedFilter/blob/master/GuidedFilteringLayer/GuidedFilter_PyTorch/guided_filter_pytorch/guided_filter.py
    """

    def __init__(self, radius: int, eps: float = 1e-8, downscale: int = 8):
        super().__init__()
        self.radius     = radius
        self.eps        = eps
        self.downscale  = downscale
        self.box_filter = BoxFilter(radius=self.radius)

    @dispatch
    def forward(self, x_lr: torch.Tensor, y_lr: torch.Tensor, x_hr: torch.Tensor) -> torch.Tensor:
        """Filters high-res image using low-res image and guide.

        Args:
            x_lr: Low-res input image as ``torch.Tensor`` in [B, C, H, W] format.
            y_lr: Low-res guidance image, same shape as ``x_lr``.
            x_hr: High-res input image, larger than ``x_lr``.
        
        Returns:
            Filtered high-res image.
        
        Raises:
            AssertionError: If tensor shapes or sizes are incompatible.
        """
        n_xlr, c_xlr, h_xlr, w_xlr = x_lr.shape
        n_ylr, c_ylr, h_ylr, w_ylr = y_lr.shape
        n_xhr, c_xhr, h_xhr, w_xhr = x_hr.shape
        assert n_xlr == n_ylr and n_ylr == n_xhr
        assert c_xlr == c_xhr and (c_xlr == 1 or c_xlr == c_ylr)
        assert h_xlr == h_ylr and w_xlr == w_ylr
        assert h_xlr > 2 * self.radius + 1 and w_xlr > 2 * self.radius + 1
        N      = self.box_filter(torch.ones(1, 1, h_xlr, w_xlr, device=x_lr.device))
        mean_x = self.box_filter(x_lr) / N
        mean_y = self.box_filter(y_lr) / N
        cov_xy = self.box_filter(x_lr * y_lr) / N - mean_x * mean_y
        var_x  = self.box_filter(x_lr * x_lr) / N - mean_x * mean_x
        A      = cov_xy / (var_x + self.eps)
        b      = mean_y - A * mean_x
        mean_A = F.interpolate(A, (h_xhr, w_xhr), mode="bicubic", align_corners=True)
        mean_b = F.interpolate(b, (h_xhr, w_xhr), mode="bicubic", align_corners=True)
        return mean_A * x_hr + mean_b

    @dispatch
    def forward(self, x_lr: torch.Tensor, x_hr: torch.Tensor) -> torch.Tensor:
        """Filters high-res image with downscaled versions.

        Args:
            x_lr: Low-res input image as ``torch.Tensor`` in [B, C, H, W] format.
            x_hr: High-res input image, larger than ``x_lr``.
        
        Returns:
            Filtered high-res image.
        """
        _, _, x_h, x_w = x_lr.size()
        h    = x_h // self.downscale
        w    = x_w // self.downscale
        x_lr = geometry.resize(x_lr, (h, w), interpolation="bicubic")
        y_lr = geometry.resize(x_hr, (h, w), interpolation="bicubic")
        return self.forward(x_lr, y_lr, x_hr)


class ConvGuidedFilter(nn.Module):
    """Applies convolutional guided filtering to an image.

    Args:
        radius: Radius of filter (kernel_size = radius * 2 + 1, e.g., 1, 2, 4, 8).
            Default is ``1``.
        norm: Normalization layer. Default is ``nn.BatchNorm2d``.
        downscale: Downscale factor for low-resolution input. Default is ``8``.
    
    References:
        - https://github.com/wuhuikai/DeepGuidedFilter/blob/master/GuidedFilteringLayer/GuidedFilter_PyTorch/guided_filter_pytorch/guided_filter.py
    """

    def __init__(
        self,
        radius   : int = 1,
        norm     : nn.Module = nn.BatchNorm2d,
        downscale: int = 8
    ):
        super().__init__()
        self.box_filter = nn.Conv2d(3, 3, 3, padding=radius, dilation=radius, bias=False, groups=3)
        self.conv_a = nn.Sequential(
            nn.Conv2d(6, 32, 1, bias=False),
            norm(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 1, bias=False),
            norm(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 1, bias=False)
        )
        self.box_filter.weight.data[...] = 1.0
        self.downscale = downscale

    @dispatch
    def forward(self, x_lr: torch.Tensor, y_lr: torch.Tensor, x_hr: torch.Tensor) -> torch.Tensor:
        """Filters high-res image using low-res image and guide.

        Args:
            x_lr: Low-res input image as ``torch.Tensor`` in [B, C, H, W] format.
            y_lr: Low-res guidance image, same shape as ``x_lr``.
            x_hr: High-res input image, larger than ``x_lr``.
        
        Returns:
            Filtered high-res image.
        """
        _, _, h_lrx, w_lrx = x_lr.size()
        _, _, h_hrx, w_hrx = x_hr.size()
        N      = self.box_filter(torch.ones(1, 3, h_lrx, w_lrx, device=x_lr.device))
        mean_x = self.box_filter(x_lr) / N
        mean_y = self.box_filter(y_lr) / N
        cov_xy = self.box_filter(x_lr * y_lr) / N - mean_x * mean_y
        var_x  = self.box_filter(x_lr * x_lr) / N - mean_x * mean_x
        A      = self.conv_a(torch.cat([cov_xy, var_x], dim=1))
        b      = mean_y - A * mean_x
        mean_A = F.interpolate(A, (h_hrx, w_hrx), mode="bicubic", align_corners=True)
        mean_b = F.interpolate(b, (h_hrx, w_hrx), mode="bicubic", align_corners=True)
        return mean_A * x_hr + mean_b

    @dispatch
    def forward(self, x_lr: torch.Tensor, x_hr: torch.Tensor) -> torch.Tensor:
        """Filters high-res image with downscaled versions.

        Args:
            x_lr: Low-res input image as ``torch.Tensor`` in [B, C, H, W] format.
            x_hr: High-res input image, larger than ``x_lr``.
        
        Returns:
            Filtered high-res image.
        """
        _, _, x_h, x_w = x_lr.size()
        h    = x_h // self.downscale
        w    = x_w // self.downscale
        x_lr = geometry.resize(x_lr, (h, w), interpolation="bicubic")
        y_lr = geometry.resize(x_hr, (h, w), interpolation="bicubic")
        return self.forward(x_lr, y_lr, x_hr)
    

# ----- Deep Guided Filter -----
def weights_init_identity(m):
    """Initializes weights of a module to identity or zero.

    Args:
        m: Module to initialize (e.g., Conv, BatchNorm2d).
    
    Notes:
        For Conv layers:
            - If output channels < input channels, uses Xavier uniform initialization.
            - Otherwise, sets weights to zero with identity at center.
        For BatchNorm2d layers:
            - Sets weights to ``1.0`` and biases to ``0.0``.
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        n_out, n_in, h, w = m.weight.data.size()
        if n_out < n_in:
            init.xavier_uniform_(m.weight.data)
            return
        m.weight.data.zero_()
        ch, cw = h // 2, w // 2
        for i in range(n_in):
            m.weight.data[i, i, ch, cw] = 1.0
    elif classname.find("BatchNorm2d") != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data,   0.0)
        
        
def build_lr_net(
    in_channels : int   = 3,
    mid_channels: int   = 24,
    layers      : int   = 5,
    relu_slope  : float = 0.2,
    norm        : nn.Module = nn.BatchNorm2d  # Corrected from AdaptiveBatchNorm2d
) -> nn.Sequential:
    """Builds a low-resolution network.

    Args:
        in_channels: Number of input channels. Default is ``3``.
        mid_channels: Number of middle channels. Default is ``24``.
        layers: Number of layers. Default is ``5``.
        relu_slope: Slope of the LeakyReLU. Default is ``0.2``.
        norm: Normalization layer. Default is ``nn.BatchNorm2d``.
    
    Returns:
        Sequential network for low-resolution processing.
    """
    net = [
        nn.Conv2d(in_channels, mid_channels, 3, 1, 1, 1, bias=False),
        norm(mid_channels),
        nn.LeakyReLU(relu_slope, inplace=True),
    ]
    for l in range(1, layers):
        net += [
            nn.Conv2d(mid_channels, mid_channels, 3, 1, 2**l, 2**l, bias=False),
            norm(mid_channels),
            nn.LeakyReLU(relu_slope, inplace=True)
        ]
    net += [
        nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, 1, bias=False),
        norm(mid_channels),
        nn.LeakyReLU(relu_slope, inplace=True),
        nn.Conv2d(mid_channels, in_channels, 1, 1, 0, 1)
    ]
    net = nn.Sequential(*net)
    net.apply(weights_init_identity)
    return net


class DeepGuidedFilter(nn.Module):
    """Deep Guided Filter network.

    Args:
        radius: Radius of filter (kernel_size = radius * 2 + 1, e.g., 1, 2, 4, 8).
            Default is ``1``.
        eps: Sharpness control value. Default is ``1e-8``.
        lr_channels: Channels for low-res network. Default is ``24``.
        lr_layers: Layers for low-res network. Default is ``5``.
        lr_relu_slope: LeakyReLU slope for low-res network. Default is ``0.2``.
        lr_norm: Normalization layer for low-res network. Default is ``nn.BatchNorm2d``.
        use_guided_map: Use guided network if ``True``. Default is ``False``.
        guided_channels: Channels for guided network. Default is ``64``.
    
    References:
        - https://github.com/wuhuikai/DeepGuidedFilter/blob/master/GuidedFilteringLayer/GuidedFilter_PyTorch/guided_filter_pytorch/guided_filter.py.
    """

    def __init__(
        self,
        radius         : int   = 1,
        eps            : float = 1e-8,
        lr_channels    : int   = 24,
        lr_layers      : int   = 5,
        lr_relu_slope  : float = 0.2,
        lr_norm        : nn.Module = nn.BatchNorm2d,
        use_guided_map : bool  = False,
        guided_channels: int   = 64
    ):
        super().__init__()
        self.lr_net = build_lr_net(
            in_channels  = 3,
            mid_channels = lr_channels,
            layers       = lr_layers,
            relu_slope   = lr_relu_slope,
            norm         = lr_norm
        )
        
        if use_guided_map:
            self.guided_map_net = nn.Sequential(
                nn.Conv2d(3, guided_channels, 1, bias=False),
                nn.BatchNorm2d(guided_channels),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(guided_channels, 3, 1)
            )
            self.guided_map_net.apply(weights_init_identity)  # Corrected from guided_map
        else:
            self.guided_map_net = None
            
        self.gf = FastGuidedFilter(radius=radius, eps=eps)

    def forward(self, x_lr: torch.Tensor, x_hr: torch.Tensor) -> torch.Tensor:
        """Filters high-res image using low-res network and guided filter.

        Args:
            x_lr: Low-res input image as torch.Tensor in [B, C, H, W] format.
            x_hr: High-res input image, larger than ``x_lr``.
        
        Returns:
            Filtered high-res image, clamped to [0, 1].
        """
        if self.guided_map_net:
            return self.gf(self.guided_map_net(x_lr), self.lr_net(x_lr), self.guided_map_net(x_hr)).clamp(0, 1)
        return self.gf(x_lr, self.lr_net(x_lr), x_hr).clamp(0, 1)

    def load_lr_weights(self, path: str | core.Path):
        """Loads weights for the low-resolution network.

        Args:
            path: Path to the weights file as ``str`` or ``core.Path``.
        """
        self.lr_net.load_state_dict(torch.load(str(path)))


class DeepConvGuidedFilter(nn.Module):
    """Deep Guided Filter network with convolutional guided filter.

    Args:
        radius: Radius of filter (kernel_size = radius * 2 + 1, e.g., 1, 2, 4, 8).
            Default is ``1``.
        lr_channels: Middle channels for low-res network. Default is ``24``.
        lr_layers: Layers for low-res network. Default is ``5``.
        lr_relu_slope: LeakyReLU slope for low-res network. Default is ``0.2``.
        lr_norm: Normalization layer for low-res network. Default is ``nn.BatchNorm2d``.
        use_guided_map: Use guided network if ``True``. Default is ``False``.
        guided_channels: Channels for guided network. Default is ``64``.
        
    References:
        - https://github.com/wuhuikai/DeepGuidedFilter/blob/master/GuidedFilteringLayer/GuidedFilter_PyTorch/guided_filter_pytorch/guided_filter.py.
    """

    def __init__(
        self,
        radius         : int   = 1,
        lr_channels    : int   = 24,
        lr_layers      : int   = 5,
        lr_relu_slope  : float = 0.2,
        lr_norm        : nn.Module = nn.BatchNorm2d, # Corrected from AdaptiveBatchNorm2d
        use_guided_map : bool  = False,
        guided_channels: int   = 64,
        guided_dilation: int   = 0,
    ):
        super().__init__()
        self.lr = build_lr_net(
            in_channels  = 3,
            mid_channels = lr_channels,
            layers       = lr_layers,
            relu_slope   = lr_relu_slope,
            norm         = lr_norm
        )
        
        if use_guided_map:
            if guided_dilation == 0:
                first_conv = nn.Conv2d(3, guided_channels, 1, bias=False)
            else:
                first_conv = nn.Conv2d(
                    3, guided_channels, 5,
                    padding  = guided_dilation,
                    dilation = guided_dilation,
                    bias     = False
                )
            self.guided_map_net = nn.Sequential(
                first_conv,
                nn.AdaptiveBatchNorm2d(guided_channels),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(guided_channels, 3, 1)
            )
        else:
            self.guided_map_net = None
            
        self.gf = ConvGuidedFilter(radius=radius, norm=nn.BatchNorm2d)  # Consistent with lr_norm

    def forward(self, x_lr: torch.Tensor, x_hr: torch.Tensor) -> torch.Tensor:
        """Filters high-res image using low-res network and conv guided filter.

        Args:
            x_lr: Low-res input image as ``torch.Tensor`` in [B, C, H, W] format.
            x_hr: High-res input image, larger than ``x_lr``.
        
        Returns:
            Filtered high-res image, clamped to [0, 1].
        """
        if self.guided_map_net:
            return self.gf(self.guided_map_net(x_lr), self.lr(x_lr), self.guided_map_net(x_hr)).clamp(0, 1)
        return self.gf(x_lr, self.lr(x_lr), x_hr).clamp(0, 1)

    def init_lr(self, path: str | core.Path):
        """Loads weights for the low-resolution network.

        Args:
            path: Path to the weights file as ``str`` or ``core.Path``.
        """
        self.lr.load_state_dict(torch.load(str(path)))
