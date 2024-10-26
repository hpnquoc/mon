#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implicit Neural Function.

References:
	https://github.com/ctom2/colie
"""

from __future__ import annotations

__all__ = [
	"OutputINF",
    "PatchFFINF",
    "PatchINF",
    "SpatialFFINF",
    "SpatialINF",
]

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from mon import core
from mon.nn.modules.inr.siren import SIRENLayer


# region INF

class PatchINF(nn.Module):
    """Implicit Neural Function built on top of SIREN.
    
    References:
        https://github.com/ctom2/colie
    """
    
    def __init__(
        self,
        window_size : int   = 1,
        out_channels: int   = 256,
        down_size   : int   = 256,
        num_layers  : int   = 2,
        w0          : float = 30.0,
        C           : float = 6.0,
        weight_decay: float = 0.0001,
    ):
        super().__init__()
        self.window_size = window_size
        self.patch_dim   = window_size ** 2
        self.down_size   = down_size
        
        patch_layers     = [SIRENLayer(self.patch_dim, out_channels, w0, C, is_first=True)]
        for _ in range(1, num_layers):
            patch_layers.append(SIRENLayer(out_channels, out_channels, w0, C))
        patch_layers.append(SIRENLayer(out_channels, out_channels, w0, C))
        self.patch_net = nn.Sequential(*patch_layers)
        
        weight_decay = weight_decay or 0.0001
        self.params  = [{"params": self.patch_net.parameters(), "weight_decay": weight_decay}]
        
    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        image_lr = self.interpolate_image(image)
        patch    = self.patch_net(self.get_patches(image_lr))
        return image_lr, patch
    
    def interpolate_image(self, image: torch.Tensor) -> torch.Tensor:
        """Reshapes the image based on new resolution."""
        return F.interpolate(image, size=(self.down_size, self.down_size), mode="bicubic")
    
    def get_patches(self, image: torch.Tensor) -> torch.Tensor:
        """Creates a tensor where the channel contains patch information."""
        num_channels = core.get_image_num_channels(image)
        kernel       = torch.zeros((self.window_size ** 2, num_channels, self.window_size, self.window_size)).to(image.device)
        for i in range(self.window_size):
            for j in range(self.window_size):
                kernel[int(torch.sum(kernel).item()), 0, i, j] = 1
        
        pad       = nn.ReflectionPad2d(self.window_size // 2)
        im_padded = pad(image)
        extracted = F.conv2d(im_padded, kernel, padding=0).squeeze(0)
        return torch.movedim(extracted, 0, -1)


class SpatialINF(nn.Module):
    """Implicit Neural Function for coordinates built on top of SIREN.
    
    References:
        https://github.com/ctom2/colie
    """
    
    def __init__(
        self,
        out_channels: int   = 256,
        down_size   : int   = 256,
        num_layers  : int   = 2,
        w0          : float = 30.0,
        C           : float = 6.0,
        weight_decay: float = 0.1,
    ):
        super().__init__()
        self.in_channels = 2
        self.down_size   = down_size
        
        spatial_layers = [SIRENLayer(self.in_channels, out_channels, w0, C, is_first=True)]
        for _ in range(1, num_layers):
            spatial_layers.append(SIRENLayer(out_channels, out_channels, w0, C))
        spatial_layers.append(SIRENLayer(out_channels, out_channels, w0, C))
        self.spatial_net = nn.Sequential(*spatial_layers)
        
        weight_decay = weight_decay or 0.1
        self.params  = [{"params": self.spatial_net.parameters(), "weight_decay": weight_decay}]
        
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        spatial = self.spatial_net(self.get_coords().to(image.device))
        return spatial
    
    def get_coords(self) -> torch.Tensor:
        """Creates a coordinates grid."""
        coords = np.dstack(
            np.meshgrid(
                np.linspace(0, 1, self.down_size),
                np.linspace(0, 1, self.down_size)
            )
        )
        coords = torch.from_numpy(coords).float()
        return coords


class OutputINF(nn.Module):
    """Implicit Neural Function for merging patch and spatial information built
    on top of SIREN.
    
    References:
        https://github.com/ctom2/colie
    """
    
    def __init__(
        self,
        in_channels : int   = 256,
        out_channels: int   = 3,
        num_layers  : int   = 1,
        w0          : float = 30.0,
        C           : float = 6.0,
        weight_decay: float = 0.001,
    ):
        super().__init__()
        
        output_layers = []
        for _ in range(0, num_layers):
            output_layers.append(SIRENLayer(in_channels, in_channels, w0, C))
        output_layers.append(SIRENLayer(in_channels, out_channels, w0, C, is_last=True))
        self.output_net = nn.Sequential(*output_layers)
        
        weight_decay = weight_decay or 0.001
        self.params  = [{"params": self.output_net.parameters(), "weight_decay": weight_decay}]
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.output_net(input)
    
# endregion


# region FF-INF

class PatchFFINF(nn.Module):
    """Implicit Neural Function built on top of SIREN.
    
    References:
        https://github.com/ctom2/colie
    """
    
    def __init__(
        self,
        window_size   : int   = 1,
        out_channels  : int   = 256,
        down_size     : int   = 256,
        num_layers    : int   = 2,
        w0            : float = 30.0,
        C             : float = 6.0,
        gaussian_scale: int   = 10,
        weight_decay  : float = 0.0001,
    ):
        super().__init__()
        self.window_size = window_size
        self.patch_dim   = window_size ** 2
        self.down_size   = down_size
        
        self.register_buffer("B", torch.randn((self.patch_dim, self.patch_dim)) * gaussian_scale)
        
        patch_layers = [SIRENLayer(self.patch_dim * 2, out_channels, w0, C, is_first=True)]
        for _ in range(1, num_layers):
            patch_layers.append(SIRENLayer(out_channels, out_channels, w0, C))
        patch_layers.append(SIRENLayer(out_channels, out_channels, w0, C))
        self.patch_net = nn.Sequential(*patch_layers)
        
        weight_decay = weight_decay or 0.0001
        self.params  = [{"params": self.patch_net.parameters(), "weight_decay": weight_decay}]
        
    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        image_lr  = self.interpolate_image(image)
        patch     = self.get_patches(image_lr)
        embedding = self.input_mapping(patch, self.B)
        patch     = self.patch_net(embedding)
        return image_lr, patch
    
    def interpolate_image(self, image: torch.Tensor) -> torch.Tensor:
        """Reshapes the image based on new resolution."""
        return F.interpolate(image, size=(self.down_size, self.down_size), mode="bicubic")
    
    def get_patches(self, image: torch.Tensor) -> torch.Tensor:
        """Creates a tensor where the channel contains patch information."""
        num_channels = core.get_image_num_channels(image)
        kernel       = torch.zeros((self.window_size ** 2, num_channels, self.window_size, self.window_size)).to(image.device)
        for i in range(self.window_size):
            for j in range(self.window_size):
                kernel[int(torch.sum(kernel).item()), 0, i, j] = 1
        
        pad       = nn.ReflectionPad2d(self.window_size // 2)
        im_padded = pad(image)
        extracted = F.conv2d(im_padded, kernel, padding=0).squeeze(0)
        return torch.movedim(extracted, 0, -1)
    
    def input_mapping(self, x: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        if B is None:
            return x
        else:
            x_proj = (2.*np.pi * x) @ B.T
            embedding = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], axis=-1)
            return embedding
        

class SpatialFFINF(nn.Module):
    """Implicit Neural Function for coordinates built on top of SIREN.
    
    References:
        https://github.com/ctom2/colie
    """
    
    def __init__(
        self,
        out_channels  : int   = 256,
        down_size     : int   = 256,
        num_layers    : int   = 2,
        w0            : float = 30.0,
        C             : float = 6.0,
        gaussian_scale: int   = 10,
        weight_decay  : float = 0.1,
    ):
        super().__init__()
        self.in_channels = 2
        self.down_size   = down_size
        
        self.register_buffer("B", torch.randn((down_size, self.in_channels)) * gaussian_scale)
        
        spatial_layers = [SIRENLayer(down_size * 2, out_channels, w0, C, is_first=True)]
        for _ in range(1, num_layers):
            spatial_layers.append(SIRENLayer(out_channels, out_channels, w0, C))
        spatial_layers.append(SIRENLayer(out_channels, out_channels, w0, C))
        self.spatial_net = nn.Sequential(*spatial_layers)
        
        weight_decay = weight_decay or 0.1
        self.params  = [{"params": self.spatial_net.parameters(), "weight_decay": weight_decay}]
        
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        coords    = self.get_coords().to(image.device)
        embedding = self.input_mapping(coords, self.B)
        spatial   = self.spatial_net(embedding)
        return spatial
    
    def get_coords(self) -> torch.Tensor:
        """Creates a coordinates grid."""
        coords = np.dstack(
            np.meshgrid(
                np.linspace(0, 1, self.down_size),
                np.linspace(0, 1, self.down_size)
            )
        )
        coords = torch.from_numpy(coords).float()
        return coords
    
    def input_mapping(self, x: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        if B is None:
            return x
        else:
            x_proj = (2.*np.pi * x) @ B.T
            embedding = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], axis=-1)
            return embedding

# endregion
