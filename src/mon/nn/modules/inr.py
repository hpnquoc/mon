#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implicit Neural Representations.

This module implements Implicit Neural Representations (INR), their layers and
networks.

References:
    https://github.com/lucidrains/siren-pytorch
    https://github.com/vishwa91/wire
"""

from __future__ import annotations

__all__ = [
    "FINER",
    "GAUSS",
    "INRLayer",
    "PEMLP",
    "SIREN",
    "WIRE",
]

from typing import Any, Literal

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from mon.nn.modules import activation as act

INR_AF = Literal["sigmoid", "tanh", "relu", "sine", "gauss", "wire", "finer"]


# region Utils

def get_image_size(x: Any) -> tuple[int, int]:
    from mon.vision.dtype import image as I
    return I.get_image_size(x)


def get_image_num_channels(image: torch.Tensor | np.ndarray) -> int:
    from mon.vision.dtype import image as I
    return I.get_image_num_channels(image)


def get_coords(down_size: int) -> torch.Tensor:
    """Creates a coordinates grid.
    
    Args:
        down_size: The size of the coordinates grid.
    """
    h, w   = down_size, down_size
    coords = np.dstack(np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w)))
    coords = torch.from_numpy(coords).float()
    return coords


def get_patches(image: torch.Tensor, kernel_size: int = 1) -> torch.Tensor:
    """Creates a tensor where the channel contains patch information."""
    from mon.vision.dtype import image as I
    num_channels = I.get_image_num_channels(image)
    kernel       = torch.zeros((kernel_size ** 2, num_channels, kernel_size, kernel_size)).to(image.device)
    for i in range(kernel_size):
        for j in range(kernel_size):
            kernel[int(torch.sum(kernel).item()), 0, i, j] = 1
    
    pad       = nn.ReflectionPad2d(kernel_size // 2)
    im_padded = pad(image)
    extracted = F.conv2d(im_padded, kernel, padding=0).squeeze(0)
    return torch.movedim(extracted, 0, -1)


def interpolate_image(image: torch.Tensor, down_size: int) -> torch.Tensor:
    """Reshapes the image based on new resolution."""
    return F.interpolate(image, size=(down_size, down_size), mode="bicubic")


def ff_embedding(p: torch.Tensor, B: torch.Tensor = None) -> torch.Tensor:
    if B is None:
        return p
    else:
        x_proj    = (2. * np.pi * p) @ B.T
        embedding = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], axis=-1)
        return embedding

# endregion


# region INR Activation Layers

class SigmoidLayer(nn.Module):
    """Drop in replacement for SineLayer but with Sigmoid non-linearity.
    
    Args:
        in_channels: The number of input channels.
        out_channels: The number of output channels.
        bias: Whether to use bias. Defaults: ``True``.
    """
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        bias        : bool = True,
        *args, **kwargs
    ):
        super().__init__()
        self.in_channels = in_channels
        self.linear      = nn.Linear(in_channels, out_channels, bias)
        self.act         = act.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.linear(x))
    

class TanhLayer(nn.Module):
    """Drop in replacement for SineLayer but with Tanh non-linearity.
    
    Args:
        in_channels: The number of input channels.
        out_channels: The number of output channels.
        bias: Whether to use bias. Defaults: ``True``.
    """
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        bias        : bool = True,
        *args, **kwargs
    ):
        super().__init__()
        self.in_channels = in_channels
        self.linear      = nn.Linear(in_channels, out_channels, bias)
        self.act         = act.Tanh()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.linear(x))
    
    
class ReLULayer(nn.Module):
    """Drop in replacement for SineLayer but with ReLU non-linearity
    
    Args:
        in_channels: The number of input channels.
        out_channels: The number of output channels.
        bias: Whether to use bias. Defaults: ``True``.
    
    References:
        https://github.com/vishwa91/wire/blob/main/modules/relu.py
    """
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        bias        : bool  = True,
        *args, **kwargs
    ):
        super().__init__()
        self.in_channels = in_channels
        self.linear      = nn.Linear(in_channels, out_channels, bias=bias)
        self.act         = act.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.linear(x))
 

class SineLayer(nn.Module):
    """Sine Layer.
    
    See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for
    discussion of omega_0.
    
    Args:
        in_channels: The number of input channels.
        out_channels: The number of output channels.
        omega_0: The frequency of the sine activation function. Defaults: ``30.0``.
        is_first: Whether this is the first layer. Defaults: ``False``.
        bias: Whether to use bias. Defaults: ``True``.
        init_weights: Whether to initialize the weights. Defaults: ``True``.
    
    References:
        https://github.com/vishwa91/wire/blob/main/modules/siren.py
    """
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        omega_0     : float = 30.0,
        is_first    : bool  = False,
        bias        : bool  = True,
        init_weights: bool  = True,
        *args, **kwargs
    ):
        super().__init__()
        self.in_channels = in_channels
        self.omega_0     = omega_0
        self.is_first    = is_first
        self.linear      = nn.Linear(in_channels, out_channels, bias)
        if init_weights:
            self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_channels, 1 / self.in_channels)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_channels) / self.omega_0,
                                             np.sqrt(6 / self.in_channels) / self.omega_0)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * self.linear(x))

    def forward_with_intermediate(self, x: torch.Tensor) -> torch.Tensor:
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(x)
        return torch.sin(intermediate), intermediate


class GaussLayer(nn.Module):
    """Drop in replacement for SineLayer but with Gaussian non-linearity
    
    Args:
        in_channels: The number of input channels.
        out_channels: The number of output channels.
        scale: The scale factor. Defaults: ``10.0``.
        bias: Whether to use bias. Defaults: ``True``.
    
    References:
        https://github.com/vishwa91/wire/blob/main/modules/gauss.py
    """
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        scale       : float = 10.0,
        bias        : bool  = True,
        *args, **kwargs
    ):
        super().__init__()
        self.in_channels = in_channels
        self.scale       = scale
        self.linear      = nn.Linear(in_channels, out_channels, bias=bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(-(self.scale * self.linear(x)) ** 2)


class FINERLayer(nn.Module):
    """FINER Layer.
    
    For the value of ``first_bias_scale``, see Fig. 5 in the paper.
    
    Args:
        in_channels: The number of input channels.
        out_channels: The number of output channels.
        omega_0: The frequency of the sine activation function. Defaults: ``30.0``.
        first_bias_scale: The scale of the first bias. Defaults: ``20.0``.
        bias: Whether to use bias. Defaults: ``True``.
        is_first: Whether this is the first layer. Defaults: ``False``.
        scale_req_grad: Whether the scale requires gradient. Defaults: ``False``.
    
    References:
        https://github.com/liuzhen0212/FINER/blob/main/models.py
    """
    
    def __init__(
        self,
        in_channels     : int,
        out_channels    : int,
        omega_0         : float = 30.0,
        first_bias_scale: float = 20.0,
        is_first        : bool  = False,
        bias            : bool  = True,
        scale_req_grad  : bool  = False,
    ):
        super().__init__()
        self.omega_0     = omega_0
        self.is_first    = is_first
        self.in_channels = in_channels
        self.linear      = nn.Linear(in_channels, out_channels, bias)
        
        self.init_weights()
        self.scale_req_grad   = scale_req_grad
        self.first_bias_scale = first_bias_scale
        if self.first_bias_scale is not None:
            self.init_first_bias()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_channels, 1 / self.in_channels)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_channels) / self.omega_0,
                                             np.sqrt(6 / self.in_channels) / self.omega_0)

    def init_first_bias(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.bias.uniform_(-self.first_bias_scale, self.first_bias_scale)
    
    def generate_scale(self, x: torch.Tensor) -> torch.Tensor:
        if self.scale_req_grad:
            scale = torch.abs(x) + 1
        else:
            with torch.no_grad():
                scale = torch.abs(x) + 1
        return scale
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        linear = self.linear(x)
        scale  = self.generate_scale(linear)
        return torch.sin(self.omega_0 * scale * linear)


class ComplexGaborLayer(nn.Module):
    """Complex Gabor Layer from WIRE (https://github.com/vishwa91/wire)
    
    References:
        https://github.com/liuzhen0212/FINER/blob/main/models.py
    """
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        omega_0     : float = 10.0,
        sigma_0     : float = 40.0,
        is_first    : bool  = False,
        bias        : bool  = True,
        trainable   : bool  = False,
    ):
        super().__init__()
        self.omega_0     = omega_0
        self.scale_0     = sigma_0
        self.is_first    = is_first
        self.in_channels = in_channels
        
        if self.is_first:
            dtype = torch.float
        else:
            dtype = torch.cfloat
            
        # Set trainable parameters if they are to be simultaneously optimized
        self.omega_0 = nn.Parameter(self.omega_0 * torch.ones(1), trainable)
        self.scale_0 = nn.Parameter(self.scale_0 * torch.ones(1), trainable)
        self.linear  = nn.Linear(in_channels, out_channels, bias=bias, dtype=dtype)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        linear = self.linear(x)
        omega  = self.omega_0 * linear
        scale  = self.scale_0 * linear
        return torch.exp(1j * omega - scale.abs().square())


class PositionalEncodingLayer(nn.Module):
    """Layer used in PEMLP.
    
    References:
        https://github.com/liuzhen0212/FINER/blob/main/models.py
    """
    
    def __init__(
        self,
        in_channels: int,
        N_freqs    : int,
        logscale   : bool = True,
    ):
        super().__init__()
        self.N_freqs      = N_freqs
        self.in_channels  = in_channels
        self.funcs        = [torch.sin, torch.cos]
        self.out_channels = in_channels * (len(self.funcs) * N_freqs + 1)

        if logscale:
            self.freq_bands = 2 ** torch.linspace(0, N_freqs - 1, N_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2 ** (N_freqs - 1), N_freqs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = [x]
        for freq in self.freq_bands:
            for func in self.funcs:
                out += [func(freq * x)]
        return torch.cat(out, -1)


class INRLayer(nn.Module):
    """INR Layer with different nonlinear layers. The layer consists of:
    (linear + activation function) + dropout.
    
    Args:
        in_channels: The number of input channels.
        out_channels: The number of output channels.
        nonlinear: The non-linearity to use. The layer defined here already
            include a ``nn.Linear()`` layer. One of: ``"sigmoid"``, ``"tanh"``,
            ``"relu"``, ``"sine"``, ``"gauss"``, ``"finer"``, ``"wire"``.
            Defaults: ``"sine"``.
        omega_0: The frequency of the sine activation function. Defaults: ``30.0``.
        scale: The scale factor. Defaults: ``10.0``.
        first_bias_scale: The scale of the first bias. Defaults: ``20.0``.
        is_first: Whether this is the first layer. Defaults: ``False``.
        is_last: Whether this is the last layer. Defaults: ``False``.
        bias: Whether to use bias. Defaults: ``True``.
        dropout: The dropout rate. Defaults: ``0.0``.
    """
    
    def __init__(
        self,
        in_channels     : int,
        out_channels    : int,
        nonlinear       : Literal["sigmoid", "tanh", "relu", "sine", "gauss", "wire", "finer"] = "sine",
        omega_0         : float = 30.0,
        scale           : float = 10.0,
        first_bias_scale: float = None,
        is_first        : bool  = False,
        is_last         : bool  = False,
        bias            : bool  = True,
        dropout         : float = 0.0,
    ):
        super().__init__()
        if is_last:
            nonlinear = "sigmoid"
            
        if nonlinear == "sigmoid":
            self.nonlinear = SigmoidLayer(
                in_channels  = in_channels,
                out_channels = out_channels,
                bias         = bias,
            )
        elif nonlinear == "tanh":
            self.nonlinear = TanhLayer(
                in_channels  = in_channels,
                out_channels = out_channels,
                bias         = bias,
            )
        elif nonlinear == "relu":
            self.nonlinear = ReLULayer(
                in_channels  = in_channels,
                out_channels = out_channels,
                bias         = bias,
            )
        elif nonlinear == "sine":
            self.nonlinear = SineLayer(
                in_channels  = in_channels,
                out_channels = out_channels,
                omega_0      = omega_0,
                is_first     = is_first,
                bias         = bias,
                init_weights = not is_last,
            )
        elif nonlinear == "gauss":
            self.nonlinear = GaussLayer(
                in_channels  = in_channels,
                out_channels = out_channels,
                scale        = scale,
                bias         = bias,
            )
        elif nonlinear == "wire":
            self.nonlinear = ComplexGaborLayer(
                in_channels  = in_channels,
                out_channels = out_channels,
                omega_0      = omega_0,
                is_first     = is_first,
                bias         = bias,
            )
        elif nonlinear == "finer":
            self.nonlinear = FINERLayer(
                in_channels      = in_channels,
                out_channels     = out_channels,
                omega_0          = omega_0,
                first_bias_scale = first_bias_scale,
                is_first         = is_first,
                bias             = bias,
                scale_req_grad   = False,
            )
        else:
            raise ValueError(f"Non-linearity '{nonlinear}' is not supported.")
            
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.nonlinear(x)
        y = self.dropout(y)
        return y

# endregion


# region INR Networks

class SIREN(nn.Module):
    """SIREN network.
    
    References:
        https://github.com/liuzhen0212/FINER/blob/main/models.py
    """
    
    def __init__(
        self,
        in_channels    : int,
        out_channels   : int,
        hidden_channels: int,
        hidden_layers  : int,
        first_omega_0  : float = 30.0,
        hidden_omega_0 : float = 30.0,
        bias           : bool  = True,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.hidden_layers   = hidden_layers
        
        net = [SineLayer(in_channels, hidden_channels, first_omega_0, is_first=True, bias=bias)]
        for i in range(hidden_layers):
            net.append(SineLayer(hidden_channels, hidden_channels, hidden_omega_0, bias=bias,))
        
        final_linear = nn.Linear(hidden_channels, out_channels)
        with torch.no_grad():
            final_linear.weight.uniform_(-np.sqrt(6 / hidden_channels) / hidden_omega_0,
                                          np.sqrt(6 / hidden_channels) / hidden_omega_0)
        net.append(final_linear)
        self.net = nn.Sequential(*net)
    
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        h, w   = get_image_size(image)
        coords = get_coords((h, w)).to(image.device)
        return self.net(coords)


class GAUSS(nn.Module):
    """Gauss network.
    
    References:
        https://github.com/liuzhen0212/FINER/blob/main/models.py
    """
    
    def __init__(
        self,
        in_channels    : int,
        out_channels   : int,
        hidden_channels: int,
        hidden_layers  : int,
        scale          : float = 30.0,
        bias           : bool  = True,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.hidden_layers   = hidden_layers
        
        net = [GaussLayer(in_channels, hidden_channels, scale, bias=bias)]
        for i in range(hidden_layers):
            net.append(GaussLayer(hidden_channels, hidden_channels, scale, bias=bias))
        final_linear = nn.Linear(hidden_channels, out_channels)
        net.append(final_linear)
        self.net = nn.Sequential(*net)
    
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        h, w   = get_image_size(image)
        coords = get_coords((h, w)).to(image.device)
        return self.net(coords)


class FINER(nn.Module):
    """FINER network.
    
    References:
        https://github.com/liuzhen0212/FINER/blob/main/models.py
    """
    
    def __init__(
        self,
        in_channels     : int,
        out_channels    : int,
        hidden_channels : int,
        hidden_layers   : int,
        first_omega_0   : float = 30.0,
        hidden_omega_0  : float = 30.0,
        first_bias_scale: float = None,
        bias            : bool  = True,
        scale_req_grad  : bool  = False
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.hidden_layers   = hidden_layers
        
        net = [FINERLayer(in_channels, hidden_channels, first_omega_0, first_bias_scale, is_first=True, bias=bias, scale_req_grad=scale_req_grad)]
        for i in range(hidden_layers):
            net.append(FINERLayer(hidden_channels, hidden_channels, hidden_omega_0, bias=bias, scale_req_grad=scale_req_grad))
        
        final_linear = nn.Linear(hidden_channels, out_channels)
        with torch.no_grad():
            final_linear.weight.uniform_(-np.sqrt(6 / hidden_channels) / hidden_omega_0,
                                          np.sqrt(6 / hidden_channels) / hidden_omega_0)
        net.append(final_linear)
        self.net = nn.Sequential(*net)
    
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        h, w   = get_image_size(image)
        coords = get_coords((h, w)).to(image.device)
        return self.net(coords)


class WIRE(nn.Module):
    """WIRE network.
    
    References:
        https://github.com/liuzhen0212/FINER/blob/main/models.py
    """
    
    def __init__(
        self,
        in_channels    : int,
        out_channels   : int,
        hidden_channels: int,
        hidden_layers  : int,
        first_omega_0  : float = 20,
        hidden_omega_0 : float = 20,
        scale          : float = 10.0,
        bias           : bool  = True,
    ):
        super().__init__()
        # Since complex numbers are two real numbers, reduce the number of hidden parameters by 2
        hidden_channels = int(hidden_channels / np.sqrt(2))
        dtype           = torch.cfloat
        
        self.hidden_channels = hidden_channels
        self.hidden_layers   = hidden_layers
        
        net = [ComplexGaborLayer(in_channels, hidden_channels, first_omega_0, sigma_0=scale, is_first=True, bias=bias)]
        for i in range(hidden_layers):
            net.append(ComplexGaborLayer(hidden_channels, hidden_channels, hidden_omega_0, sigma_0=scale, bias=bias))
        
        final_linear = nn.Linear(hidden_channels, out_channels, dtype=dtype)
        with torch.no_grad():
            final_linear.weight.uniform_(-np.sqrt(6 / hidden_channels) / hidden_omega_0,
                                          np.sqrt(6 / hidden_channels) / hidden_omega_0)
        net.append(final_linear)
        self.net = nn.Sequential(*net)
    
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        h, w   = get_image_size(image)
        coords = get_coords((h, w)).to(image.device)
        return self.net(coords)


class PEMLP(nn.Module):
    
    def __init__(
        self,
        in_channels    : int,
        out_channels   : int,
        hidden_channels: int,
        hidden_layers  : int,
        N_freqs        : int = 10,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.hidden_layers   = hidden_layers
        self.encoding        = PositionalEncodingLayer(in_channels=in_channels, N_freqs=N_freqs)
        
        self.net = []
        self.net.append(nn.Linear(self.encoding.out_channels, hidden_channels))
        self.net.append(nn.ReLU(True))

        for i in range(hidden_layers):
            self.net.append(nn.Linear(hidden_channels, hidden_channels))
            self.net.append(nn.ReLU(True))

        final_linear = nn.Linear(hidden_channels, out_channels)
        self.net.append(final_linear)
        self.net = nn.Sequential(*self.net)
    
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        h, w   = get_image_size(image)
        coords = get_coords((h, w)).to(image.device)
        return self.net(self.enconding(coords))

# endregion
