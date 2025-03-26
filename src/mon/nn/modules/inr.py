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

INR_AF = Literal["sigmoid", "tanh", "relu", "sine", "gauss", "wire", "finer"]


# region Utils

def get_size(x: Any) -> tuple[int, int]:
    """Gets the size of an image as ``(height, width)``.

    Args:
        x: Image or data to measure.
    
    Returns:
        Tuple of ``(height, width)`` in pixels.
    """
    from mon.vision.dtype import image as I
    return I.get_image_size(x)


def get_image_num_channels(image: torch.Tensor | np.ndarray) -> int:
    """Gets the number of channels in an image.

    Args:
        image: Tensor or array representing an image.
    
    Returns:
        Number of channels as an integer.
    """
    from mon.vision.dtype import image as I
    return I.get_image_num_channels(image)


def get_coords(down_size: int) -> torch.Tensor:
    """Creates a coordinates grid.

    Args:
        down_size: Size of the square coordinates grid.
    
    Returns:
        Tensor of shape ``(down_size, down_size, 2)`` with normalized coords.
    """
    h, w   = down_size, down_size
    coords = np.dstack(np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w)))
    return torch.from_numpy(coords).float()


def get_patches(image: torch.Tensor, kernel_size: int = 1) -> torch.Tensor:
    """Extracts patches into channels of a tensor.

    Args:
        image: Tensor of shape ``(C, H, W)`` or ``(B, C, H, W)``.
        kernel_size: Size of square patches. Default is ``1``.
    
    Returns:
        Tensor with patches in channels, shape ``(H', W', K^2)`` or ``(B, H', W', K^2)``.
    """
    from mon.vision.dtype import image as I
    num_channels = I.get_image_num_channels(image)
    kernel       = torch.zeros(kernel_size**2, num_channels, kernel_size, kernel_size, device=image.device)
    
    for i in range(kernel_size):
        for j in range(kernel_size):
            kernel[i * kernel_size + j, :, i, j] = 1
    
    im_padded = nn.ReflectionPad2d(kernel_size // 2)(image)
    extracted = F.conv2d(im_padded, kernel, padding=0)
    return torch.movedim(extracted, 1 if image.dim() == 4 else 0, -1)


def interpolate_image(image: torch.Tensor, down_size: int) -> torch.Tensor:
    """Resizes image to a new square resolution.

    Args:
        image: Tensor of shape ``(C, H, W)`` or ``(B, C, H, W)``.
        down_size: Target size for height and width.
    
    Returns:
        Resized tensor of shape (``C, down_size, down_size)`` or ``(B, C, down_size, down_size)``.
    """
    return F.interpolate(image, size=(down_size, down_size), mode="bicubic")


def ff_embedding(p: torch.Tensor, B: torch.Tensor = None) -> torch.Tensor:
    """Applies Fourier feature embedding to input tensor.

    Args:
        p: Input tensor to embed.
        B: Projection matrix. Default is ``None``.
    
    Returns:
        Embedded tensor with sine and cosine features.
    """
    if B is None:
        return p
    x_proj = (2 * np.pi * p) @ B.T
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

# endregion


# region INR Activation Layers

class SigmoidLayer(nn.Module):
    """Applies linear transformation with sigmoid activation.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        bias: Use bias in linear layer if ``True``. Default is ``True``.
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
        self.linear      = nn.Linear(in_channels, out_channels, bias=bias)
        self.act         = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transforms input with linear layer and sigmoid.

        Args:
            x: Input tensor.
        
        Returns:
            Transformed tensor.
        """
        return self.act(self.linear(x))
    

class TanhLayer(nn.Module):
    """Applies linear transformation with tanh activation.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        bias: Use bias in linear layer if ``True``. Default is ``True``.
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
        self.linear      = nn.Linear(in_channels, out_channels, bias=bias)
        self.act         = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transforms input with linear layer and tanh.

        Args:
            x: Input tensor.
            
        Returns:
            Transformed tensor.
        """
        return self.act(self.linear(x))
    
    
class ReLULayer(nn.Module):
    """Applies linear transformation with ReLU activation.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        bias: Use bias in linear layer if ``True``. Default is ``True``.
   
    References:
        - https://github.com/vishwa91/wire/blob/main/modules/relu.py
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
        self.linear      = nn.Linear(in_channels, out_channels, bias=bias)
        self.act         = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transforms input with linear layer and ReLU.

        Args:
            x: Input tensor.
            
        Returns:
            Transformed tensor.
        """
        return self.act(self.linear(x))
 

class SineLayer(nn.Module):
    """Applies linear transformation with sine activation.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        w0: Sine frequency factor. Default is ``30.0``.
        is_first: First layer flag for weight init. Default is ``False``.
        bias: Use bias in linear layer if ``True``. Default is ``True``.
        init_weights: Initialize weights if ``True``. Default is ``True``.
    
    References:
        - https://github.com/vishwa91/wire/blob/main/modules/siren.py
    """

    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        w0          : float = 30.0,
        is_first    : bool  = False,
        bias        : bool  = True,
        init_weights: bool  = True,
        *args, **kwargs
    ):
        super().__init__()
        self.in_channels = in_channels
        self.w0          = w0
        self.is_first    = is_first
        self.linear      = nn.Linear(in_channels, out_channels, bias=bias)
        if init_weights:
            self.init_weights()

    def init_weights(self):
        """Initializes linear layer weights based on layer position."""
        with torch.no_grad():
            bound = 1 / self.in_channels if self.is_first else np.sqrt(6 / self.in_channels) / self.w0
            self.linear.weight.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transforms input with linear layer and sine.

        Args:
            x: Input tensor.
        
        Returns:
            Sine-transformed tensor.
        """
        return torch.sin(self.w0 * self.linear(x))

    def forward_with_intermediate(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Transforms input and returns intermediate result.

        Args:
            x: Input tensor.
        
        Returns:
            Tuple of (sine-transformed tensor, intermediate tensor).
        """
        intermediate = self.w0 * self.linear(x)
        return torch.sin(intermediate), intermediate


class GaussLayer(nn.Module):
    """Applies linear transformation with Gaussian activation.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        scale: Gaussian scale factor. Default is ``10.0``.
        bias: Use bias in linear layer if ``True``. Default is ``True``.
    
    References:
        - https://github.com/vishwa91/wire/blob/main/modules/gauss.py
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
        """Transforms input with linear layer and Gaussian.

        Args:
            x: Input tensor.
        
        Returns:
            Gaussian-transformed tensor.
        """
        return torch.exp(-(self.scale * self.linear(x))**2)


class FINERLayer(nn.Module):
    """Applies scaled sine activation to linear transformation.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        w0: Sine frequency factor. Default is ``30.0``.
        first_bias_scale: Bias scale for first layer. Default is ``20.0``.
        is_first: First layer flag for init. Default is ``False``.
        bias: Use bias in linear layer if ``True``. Default is ``True``.
        scale_req_grad: Scale requires gradient if ``True``. Default is ``False``.
    
    References:
        - https://github.com/liuzhen0212/FINER/blob/main/models.py
    """

    def __init__(
        self,
        in_channels     : int,
        out_channels    : int,
        w0              : float = 30.0,
        first_bias_scale: float = 20.0,
        is_first        : bool  = False,
        bias            : bool  = True,
        scale_req_grad  : bool  = False
    ):
        super().__init__()
        self.w0               = w0
        self.is_first         = is_first
        self.in_channels      = in_channels
        self.scale_req_grad   = scale_req_grad
        self.first_bias_scale = first_bias_scale
        self.linear           = nn.Linear(in_channels, out_channels, bias=bias)
        self.init_weights()
        if self.first_bias_scale is not None and self.is_first:
            self.init_first_bias()

    def init_weights(self):
        """Initializes linear layer weights based on layer position."""
        with torch.no_grad():
            bound = 1 / self.in_channels if self.is_first else np.sqrt(6 / self.in_channels) / self.w0
            self.linear.weight.uniform_(-bound, bound)

    def init_first_bias(self):
        """Initializes bias for the first layer."""
        with torch.no_grad():
            self.linear.bias.uniform_(-self.first_bias_scale, self.first_bias_scale)

    def generate_scale(self, x: torch.Tensor) -> torch.Tensor:
        """Generates scaling factor for activation.

        Args:
            x: Input tensor for scaling.
        
        Returns:
            Scaling tensor.
        """
        if self.scale_req_grad:
            return torch.abs(x) + 1
        with torch.no_grad():
            return torch.abs(x) + 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transforms input with scaled sine activation.

        Args:
            x: Input tensor.
        
        Returns:
            Transformed tensor.
        """
        linear = self.linear(x)
        scale  = self.generate_scale(linear)
        return torch.sin(self.w0 * scale * linear)


class ComplexGaborLayer(nn.Module):
    """Applies complex Gabor transformation to input.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        w0: Base frequency factor. Default is ``10.0``.
        s0: Base scale factor. Default is ``40.0``.
        is_first: First layer flag for dtype. Default is ``False``.
        bias: Use bias in linear layer if ``True``. Default is ``True``.
        trainable: Parameters trainable if ``True``. Default is ``False``.
    
    References:
        - https://github.com/vishwa91/wire
        - https://github.com/liuzhen0212/FINER/blob/main/models.py
    """

    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        w0          : float = 10.0,
        s0          : float = 40.0,
        is_first    : bool  = False,
        bias        : bool  = True,
        trainable   : bool  = False
    ):
        super().__init__()
        self.is_first    = is_first
        self.in_channels = in_channels
        dtype            = torch.float if is_first else torch.cfloat
        self.linear      = nn.Linear(in_channels, out_channels, bias=bias, dtype=dtype)
        self.w0          = nn.Parameter(torch.tensor([w0]), requires_grad=trainable)
        self.scale_0     = nn.Parameter(torch.tensor([s0]), requires_grad=trainable)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transforms input with complex Gabor activation.

        Args:
            x: Input tensor.
        
        Returns:
            Complex-valued transformed tensor.
        """
        linear = self.linear(x)
        omega  = self.w0 * linear
        scale  = self.scale_0 * linear
        return torch.exp(1j * omega - scale.abs().square())


class PositionalEncodingLayer(nn.Module):
    """Applies positional encoding with sine and cosine functions.

    Args:
        in_channels: Number of input channels.
        N_freqs: Number of frequency bands.
        logscale: Use logarithmic frequency scale if ``True``. Default is ``True``.
   
    References:
        - https://github.com/liuzhen0212/FINER/blob/main/models.py
    """

    def __init__(
        self,
        in_channels: int,
        N_freqs    : int,
        logscale   : bool = True
    ):
        super().__init__()
        self.N_freqs      = N_freqs
        self.in_channels  = in_channels
        self.funcs        = [torch.sin, torch.cos]
        self.out_channels = in_channels * (len(self.funcs) * N_freqs + 1)
        self.freq_bands   = (
            2 ** torch.linspace(0, N_freqs - 1, N_freqs) if logscale
            else torch.linspace(1, 2 ** (N_freqs - 1), N_freqs)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encodes input with positional frequency bands.

        Args:
            x: Input tensor of shape ``(..., in_channels)``.
            
        Returns:
            Encoded tensor of shape ``(..., out_channels)``.
        """
        out = [x]
        for freq in self.freq_bands:
            for func in self.funcs:
                out.append(func(freq * x))
        return torch.cat(out, dim=-1)


class INRLayer(nn.Module):
    """Combines linear transformation, nonlinearity, and dropout.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        nonlinear: Nonlinearity type: ``"sigmoid"``, ``"tanh"``, ``"relu"``,
            ``"sine"``, ``"gauss"``, ``"wire"``, ``"finer"``. Default is ``"sine"``.
        w0: Sine frequency factor. Default is ``30.0``.
        scale: Gaussian scale factor. Default is ``10.0``.
        first_bias_scale: Bias scale for first ``"finer"`` layer. Default is ``None``.
        is_first: First layer flag. Default is ``False``.
        is_last: Last layer flag, forces "sigmoid". Default is ``False``.
        bias: Use bias in linear layer if ``True``. Default is ``True``.
        dropout: Dropout rate. Default is ``0.0``.
    """
    
    def __init__(
        self,
        in_channels     : int,
        out_channels    : int,
        nonlinear       : Literal["sigmoid", "tanh", "relu", "sine", "gauss", "wire", "finer"] = "sine",
        w0              : float = 30.0,
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
        
        layer_args = {
            "in_channels" : in_channels,
            "out_channels": out_channels,
            "bias"        : bias
        }
        
        if nonlinear == "sigmoid":
            self.nonlinear = SigmoidLayer(**layer_args)
        elif nonlinear == "tanh":
            self.nonlinear = TanhLayer(**layer_args)
        elif nonlinear == "relu":
            self.nonlinear = ReLULayer(**layer_args)
        elif nonlinear == "sine":
            self.nonlinear = SineLayer(**layer_args, w0=w0, is_first=is_first, init_weights=not is_last)
        elif nonlinear == "gauss":
            self.nonlinear = GaussLayer(**layer_args, scale=scale)
        elif nonlinear == "wire":
            self.nonlinear = ComplexGaborLayer(**layer_args, w0=w0, is_first=is_first)
        elif nonlinear == "finer":
            self.nonlinear = FINERLayer(**layer_args, w0=w0, first_bias_scale=first_bias_scale, is_first=is_first)
        else:
            raise ValueError(f"[nonlinear] must be supported type, but got [{nonlinear}]")
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies nonlinearity and dropout to input.

        Args:
            x: Input tensor.
        
        Returns:
            Transformed tensor.
        """
        return self.dropout(self.nonlinear(x))

# endregion


# region INR Networks

class SIREN(nn.Module):
    """Implements SIREN network with sine layers.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        hidden_channels: Number of channels in hidden layers.
        hidden_layers: Number of hidden layers.
        first_w0: Frequency for first layer. Default is ``30.0``.
        hidden_w0: Frequency for hidden layers. Default is ``30.0``.
        bias: Use bias in layers if ``True``. Default is ``True``.
    
    References:
        - https://github.com/liuzhen0212/FINER/blob/main/models.py
    """
    
    def __init__(
        self,
        in_channels    : int,
        out_channels   : int,
        hidden_channels: int,
        hidden_layers  : int,
        first_w0       : float = 30.0,
        hidden_w0      : float = 30.0,
        bias           : bool  = True,
    ):
        super().__init__()
        self.net = nn.Sequential(
            SineLayer(in_channels, hidden_channels, first_w0, is_first=True, bias=bias),
            *[SineLayer(hidden_channels, hidden_channels, hidden_w0, bias=bias) for _ in range(hidden_layers)],
            nn.Linear(hidden_channels, out_channels)
        )
        with torch.no_grad():
            self.net[-1].weight.uniform_(-np.sqrt(6 / hidden_channels) / hidden_w0, np.sqrt(6 / hidden_channels) / hidden_w0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Generates output from image coordinates.

        Args:
            x: Input image tensor for size reference.
       
        Returns:
            Output tensor from network.
        """
        s, _   = get_size(x)
        coords = get_coords(s).to(x.device)
        return self.net(coords)


class GAUSS(nn.Module):
    """Implements Gaussian network with Gauss layers.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        hidden_channels: Number of channels in hidden layers.
        hidden_layers: Number of hidden layers.
        scale: Gaussian scale factor. Default is ``30.0``.
        bias: Use bias in layers if ``True``. Default is ``True``.
    
    References:
        - https://github.com/liuzhen0212/FINER/blob/main/models.py
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
        layers  = [GaussLayer(in_channels, hidden_channels, scale, bias=bias)]
        layers += [GaussLayer(hidden_channels, hidden_channels, scale, bias=bias) for _ in range(hidden_layers)]
        layers.append(nn.Linear(hidden_channels, out_channels))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Generates output from image coordinates.

        Args:
            x: Input image tensor for size reference.
        
        Returns:
            Output tensor from network.
        """
        s, _   = get_size(x)
        coords = get_coords(s).to(x.device)
        return self.net(coords)


class FINER(nn.Module):
    """Implements FINER network with FINER layers.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        hidden_channels: Number of channels in hidden layers.
        hidden_layers: Number of hidden layers.
        first_w0: Frequency for first layer. Default is ``30.0``.
        hidden_w0: Frequency for hidden layers. Default is ``30.0``.
        first_bias_scale: Bias scale for first layer. Default is ``None``.
        bias: Use bias in layers if ``True``. Default is ``True``.
        scale_req_grad: Scale requires gradient if ``True``. Default is ``False``.
    
    References:
        - https://github.com/liuzhen0212/FINER/blob/main/models.py
    """
    
    def __init__(
        self,
        in_channels     : int,
        out_channels    : int,
        hidden_channels : int,
        hidden_layers   : int,
        first_w0        : float = 30.0,
        hidden_w0       : float = 30.0,
        first_bias_scale: float = None,
        bias            : bool  = True,
        scale_req_grad  : bool  = False
    ):
        super().__init__()
        layers  = [FINERLayer(in_channels, hidden_channels, first_w0, first_bias_scale, is_first=True, bias=bias, scale_req_grad=scale_req_grad)]
        layers += [FINERLayer(hidden_channels, hidden_channels, hidden_w0, bias=bias, scale_req_grad=scale_req_grad) for _ in range(hidden_layers)]
        final_linear = nn.Linear(hidden_channels, out_channels)
        with torch.no_grad():
            final_linear.weight.uniform_(-np.sqrt(6 / hidden_channels) / hidden_w0, np.sqrt(6 / hidden_channels) / hidden_w0)
        layers.append(final_linear)
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Generates output from image coordinates.

        Args:
            x: Input image tensor for size reference.
        
        Returns:
            Output tensor from network.
        """
        s, _   = get_size(x)
        coords = get_coords(s).to(x.device)
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
        first_w0       : float = 20,
        hidden_w0      : float = 20,
        scale          : float = 10.0,
        bias           : bool  = True,
    ):
        super().__init__()
        # Since complex numbers are two real numbers, reduce the number of hidden parameters by 2
        hidden_channels = int(hidden_channels / np.sqrt(2))
        dtype = torch.cfloat

        layers  = [ComplexGaborLayer(in_channels, hidden_channels, first_w0, s0=scale, is_first=True, bias=bias)]
        layers += [ComplexGaborLayer(hidden_channels, hidden_channels, hidden_w0, s0=scale, bias=bias) for _ in range(hidden_layers)]
        
        final_linear = nn.Linear(hidden_channels, out_channels, dtype=dtype)
        with torch.no_grad():
            final_linear.weight.uniform_(-np.sqrt(6 / hidden_channels) / hidden_w0, np.sqrt(6 / hidden_channels) / hidden_w0)
        layers.append(final_linear)

        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Generates output from image coordinates.

        Args:
            x: Input image tensor for size reference.
        
        Returns:
            Output tensor from network.
        """
        s, _   = get_size(x)
        coords = get_coords(s).to(x.device)
        return self.net(coords)


class PEMLP(nn.Module):
    """Implements positional encoding MLP network.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        hidden_channels: Number of channels in hidden layers.
        hidden_layers: Number of hidden layers.
        N_freqs: Number of frequency bands for encoding. Default is ``10``.
    """
    
    def __init__(
        self,
        in_channels    : int,
        out_channels   : int,
        hidden_channels: int,
        hidden_layers  : int,
        N_freqs        : int = 10,
    ):
        super().__init__()
        self.encoding = PositionalEncodingLayer(in_channels=in_channels, N_freqs=N_freqs)
        
        layers  = [nn.Linear(self.encoding.out_channels, hidden_channels), nn.ReLU(True)]
        layers += [nn.Linear(hidden_channels, hidden_channels), nn.ReLU(True)] * hidden_layers
        layers.append(nn.Linear(hidden_channels, out_channels))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Generates output from encoded image coordinates.

        Args:
            image: Input image tensor for size reference.
            
        Returns:
            Output tensor from network.
        """
        s, _   = get_size(x)
        coords = get_coords(s).to(x.device)
        return self.net(self.enconding(coords))

# endregion
