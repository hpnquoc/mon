#!/usr/bin/env python
# -*- coding: utf-8 -*-

__all__ = [
    "FINER",
    "SIREN",
    "PE_FINER",
    "PE_SIREN",
]

from typing import Any

import torch

from mon.core import nn


# ----- Utils -----
def fd_2d(patch: torch.Tensor, depth: torch.Tensor = None, alpha: float = 8.3) -> torch.Tensor:
    if depth is None:
        return 1
    
    center_idx   = patch.shape[-1] // 2
    depth_center = depth[center_idx:center_idx + 1]  # Shape: [..., 1]
    depth_diff   = torch.abs(depth - depth_center)
    fd           = torch.exp(-alpha * depth_diff)
    return fd


def fd_3d(patch: torch.Tensor, depth: torch.Tensor = None, alpha: float = 8.3) -> torch.Tensor:
    if depth is None:
        return 1
    
    center_idx   = patch.shape[-1] // 2
    depth_center = depth[..., center_idx:center_idx + 1]  # Shape: [..., 1]
    depth_diff   = torch.abs(depth - depth_center)
    fd           = torch.exp(-alpha * depth_diff)
    return fd

   
# ----- INRs -----
class SIREN(nn.Module):

    def __init__(
        self,
        patch_dim   : int,
        hidden_dim  : int   = 256,
        num_layers  : int   = 4,
        add_layers  : int   = 2,
        alpha       : float = 8.3,
        weight_decay: Any   = None,
    ):
        super().__init__()
        self.alpha = alpha
        
        depth_layers   = [nn.SineLayer(patch_dim,   hidden_dim, is_first=True)]
        patch_layers   = [nn.SineLayer(patch_dim,   hidden_dim, is_first=True)]
        spatial_layers = [nn.SineLayer(2, hidden_dim, is_first=True)]
        for _ in range(1, add_layers - 2):
            depth_layers.append(nn.SineLayer(hidden_dim, hidden_dim))
            patch_layers.append(nn.SineLayer(hidden_dim, hidden_dim))
            spatial_layers.append(nn.SineLayer(hidden_dim, hidden_dim))
        depth_layers.append(nn.SineLayer(hidden_dim, hidden_dim // 2))
        patch_layers.append(nn.SineLayer(hidden_dim, hidden_dim // 2))
        spatial_layers.append(nn.SineLayer(hidden_dim, hidden_dim // 2))
        
        output_layers = []
        for _ in range(add_layers, num_layers - 1):
            output_layers.append(nn.SineLayer(hidden_dim, hidden_dim))
        output_layers.append(nn.Linear(hidden_dim, 1))
        output_layers.append(nn.Sigmoid())
        
        self.depth_net   = nn.Sequential(*depth_layers)
        self.patch_net   = nn.Sequential(*patch_layers)
        self.spatial_net = nn.Sequential(*spatial_layers)
        self.output_net  = nn.Sequential(*output_layers)
        
        if not weight_decay:
            weight_decay = [0.1, 0.0001, 0.001]
            
        self.params  = []
        self.params += [{"params": self.spatial_net.parameters(), "weight_decay": weight_decay[0]}]
        self.params += [{"params": self.depth_net.parameters(),   "weight_decay": weight_decay[1]}]
        self.params += [{"params": self.patch_net.parameters(),   "weight_decay": weight_decay[1]}]
        self.params += [{"params": self.output_net.parameters(),  "weight_decay": weight_decay[2]}]
        
    def forward(self, patch: torch.Tensor, spatial: torch.Tensor, depth: torch.Tensor = None) -> torch.Tensor:
        fd    = fd_3d(patch, depth, self.alpha)
        patch = patch * fd
        for d_layer, p_layer in zip(self.depth_net, self.patch_net):
            depth = d_layer(depth)
            patch = p_layer(patch) * fd_2d(patch, depth, self.alpha)
        # patch   = self.patch_net(patch)
        spatial = self.spatial_net(spatial)
        return self.output_net(torch.cat((patch, spatial), -1))
        

class FINER(nn.Module):
    
    def __init__(
        self,
        patch_dim   : int,
        hidden_dim  : int   = 256,
        num_layers  : int   = 4,
        add_layers  : int   = 2,
        alpha       : float = 8.3,
        weight_decay: Any   = None,
    ):
        super().__init__()
        self.alpha = alpha
        
        depth_layers   = [nn.FINERLayer(patch_dim,   hidden_dim, is_first=True)]
        patch_layers   = [nn.FINERLayer(patch_dim,   hidden_dim, is_first=True)]
        spatial_layers = [nn.FINERLayer(2, hidden_dim, is_first=True)]
        for _ in range(1, add_layers - 2):
            depth_layers.append(nn.FINERLayer(hidden_dim, hidden_dim))
            patch_layers.append(nn.FINERLayer(hidden_dim, hidden_dim))
            spatial_layers.append(nn.FINERLayer(hidden_dim, hidden_dim))
        depth_layers.append(nn.FINERLayer(hidden_dim, hidden_dim // 2))
        patch_layers.append(nn.FINERLayer(hidden_dim, hidden_dim // 2))
        spatial_layers.append(nn.FINERLayer(hidden_dim, hidden_dim // 2))
        
        output_layers = []
        for _ in range(add_layers, num_layers - 1):
            output_layers.append(nn.FINERLayer(hidden_dim, hidden_dim))
        output_layers.append(nn.Linear(hidden_dim, 1))
        output_layers.append(nn.Sigmoid())
        
        self.depth_net   = nn.Sequential(*depth_layers)
        self.patch_net   = nn.Sequential(*patch_layers)
        self.spatial_net = nn.Sequential(*spatial_layers)
        self.output_net  = nn.Sequential(*output_layers)

        if not weight_decay:
            weight_decay = [0.1, 0.0001, 0.001]
           
        self.params  = []
        self.params += [{"params": self.spatial_net.parameters(), "weight_decay": weight_decay[0]}]
        self.params += [{"params": self.depth_net.parameters(),   "weight_decay": weight_decay[1]}]
        self.params += [{"params": self.patch_net.parameters(),   "weight_decay": weight_decay[1]}]
        self.params += [{"params": self.output_net.parameters(),  "weight_decay": weight_decay[2]}]
        
    def forward(self, patch: torch.Tensor, spatial: torch.Tensor, depth: torch.Tensor = None) -> torch.Tensor:
        fd    = fd_3d(patch, depth, self.alpha)
        patch = patch * fd
        for d_layer, p_layer in zip(self.depth_net, self.patch_net):
            depth = d_layer(depth)
            patch = p_layer(patch) * fd_2d(patch, depth, self.alpha)
        # patch   = self.patch_net(patch)
        spatial = self.spatial_net(spatial)
        return self.output_net(torch.cat((patch, spatial), -1))


class PE_SIREN(nn.Module):

    def __init__(
        self,
        patch_dim   : int,
        hidden_dim  : int   = 256,
        num_layers  : int   = 4,
        add_layers  : int   = 2,
        alpha       : float = 8.3,
        weight_decay: Any   = None
    ):
        super().__init__()
        self.alpha     = alpha
        self.encoding  = nn.PositionalEncoding(in_features=2, N_freqs=10)
        spatial_dim    = self.encoding.out_features
        
        depth_layers   = [nn.SineLayer(patch_dim,   hidden_dim, is_first=True)]
        patch_layers   = [nn.SineLayer(patch_dim,   hidden_dim, is_first=True)]
        spatial_layers = [nn.SineLayer(spatial_dim, hidden_dim, is_first=True)]
        for _ in range(1, add_layers - 2):
            depth_layers.append(nn.SineLayer(hidden_dim, hidden_dim))
            patch_layers.append(nn.SineLayer(hidden_dim, hidden_dim))
            spatial_layers.append(nn.SineLayer(hidden_dim, hidden_dim))
        depth_layers.append(nn.SineLayer(hidden_dim, hidden_dim // 2))
        patch_layers.append(nn.SineLayer(hidden_dim, hidden_dim // 2))
        spatial_layers.append(nn.SineLayer(hidden_dim, hidden_dim // 2))
        
        output_layers = []
        for _ in range(add_layers, num_layers - 1):
            output_layers.append(nn.SineLayer(hidden_dim, hidden_dim))
        output_layers.append(nn.Linear(hidden_dim, 1))
        output_layers.append(nn.Sigmoid())
        
        self.depth_net   = nn.Sequential(*depth_layers)
        self.patch_net   = nn.Sequential(*patch_layers)
        self.spatial_net = nn.Sequential(*spatial_layers)
        self.output_net  = nn.Sequential(*output_layers)

        if not weight_decay:
            weight_decay = [0.1, 0.0001, 0.001]
           
        self.params  = []
        self.params += [{"params": self.spatial_net.parameters(), "weight_decay": weight_decay[0]}]
        self.params += [{"params": self.depth_net.parameters(),   "weight_decay": weight_decay[1]}]
        self.params += [{"params": self.patch_net.parameters(),   "weight_decay": weight_decay[1]}]
        self.params += [{"params": self.output_net.parameters(),  "weight_decay": weight_decay[2]}]
    
    def forward(self, patch: torch.Tensor, spatial: torch.Tensor, depth: torch.Tensor = None) -> torch.Tensor:
        fd    = fd_3d(patch, depth, self.alpha)
        patch = patch * fd
        for d_layer, p_layer in zip(self.depth_net, self.patch_net):
            depth = d_layer(depth)
            patch = p_layer(patch) * fd_2d(patch, depth, self.alpha)
        # patch   = self.patch_net(patch)
        spatial = self.spatial_net(self.encoding(spatial))
        return self.output_net(torch.cat((patch, spatial), -1))


class PE_FINER(nn.Module):

    def __init__(
        self,
        patch_dim   : int,
        hidden_dim  : int   = 256,
        num_layers  : int   = 4,
        add_layers  : int   = 2,
        alpha       : float = 8.3,
        weight_decay: Any   = None
    ):
        super().__init__()
        self.alpha     = alpha
        self.encoding  = nn.PositionalEncoding(in_features=2, N_freqs=10)
        spatial_dim    = self.encoding.out_features
        
        depth_layers   = [nn.FINERLayer(patch_dim,   hidden_dim, is_first=True)]
        patch_layers   = [nn.FINERLayer(patch_dim,   hidden_dim, is_first=True)]
        spatial_layers = [nn.FINERLayer(spatial_dim, hidden_dim, is_first=True)]
        for _ in range(1, add_layers - 2):
            depth_layers.append(nn.FINERLayer(hidden_dim, hidden_dim))
            patch_layers.append(nn.FINERLayer(hidden_dim, hidden_dim))
            spatial_layers.append(nn.FINERLayer(hidden_dim, hidden_dim))
        depth_layers.append(nn.FINERLayer(hidden_dim, hidden_dim // 2))
        patch_layers.append(nn.FINERLayer(hidden_dim, hidden_dim // 2))
        spatial_layers.append(nn.FINERLayer(hidden_dim, hidden_dim // 2))
        
        output_layers = []
        for _ in range(add_layers, num_layers - 1):
            output_layers.append(nn.FINERLayer(hidden_dim, hidden_dim))
        output_layers.append(nn.Linear(hidden_dim, 1))
        output_layers.append(nn.Sigmoid())
        
        self.depth_net   = nn.Sequential(*depth_layers)
        self.patch_net   = nn.Sequential(*patch_layers)
        self.spatial_net = nn.Sequential(*spatial_layers)
        self.output_net  = nn.Sequential(*output_layers)

        if not weight_decay:
            weight_decay = [0.1, 0.0001, 0.001]
           
        self.params  = []
        self.params += [{"params": self.spatial_net.parameters(), "weight_decay": weight_decay[0]}]
        self.params += [{"params": self.depth_net.parameters(),   "weight_decay": weight_decay[1]}]
        self.params += [{"params": self.patch_net.parameters(),   "weight_decay": weight_decay[1]}]
        self.params += [{"params": self.output_net.parameters(),  "weight_decay": weight_decay[2]}]
        
    def forward(self, patch: torch.Tensor, spatial: torch.Tensor, depth: torch.Tensor = None) -> torch.Tensor:
        fd    = fd_3d(patch, depth, self.alpha)
        patch = patch * fd
        for d_layer, p_layer in zip(self.depth_net, self.patch_net):
            depth = d_layer(depth)
            patch = p_layer(patch) * fd_2d(patch, depth, self.alpha)
        # patch   = self.patch_net(patch)
        spatial = self.spatial_net(self.encoding(spatial))
        return self.output_net(torch.cat((patch, spatial), -1))


# ----- Conv-INR -----
class ConvINR(nn.Module):

    def __init__(
        self,
        in_channels : int,
        hidden_dim  : int   = 256,
        kernel_size : int   = 3,
        num_layers  : int   = 4,
        add_layers  : int   = 2,
        alpha       : float = 8.3,
        weight_decay: Any   = None,
    ):
        super().__init__()
        self.alpha = alpha
        
        patch_layers   = [nn.Conv2d(in_channels, hidden_dim, kernel_size, padding=kernel_size // 2)]
        
        
        spatial_layers = [nn.SineLayer(2, hidden_dim, is_first=True)]
        for _ in range(1, add_layers - 2):
            patch_layers.append(nn.SineLayer(hidden_dim, hidden_dim))
            spatial_layers.append(nn.SineLayer(hidden_dim, hidden_dim))
        patch_layers.append(nn.SineLayer(hidden_dim, hidden_dim // 2))
        spatial_layers.append(nn.SineLayer(hidden_dim, hidden_dim // 2))
        
        output_layers = []
        for _ in range(add_layers, num_layers - 1):
            output_layers.append(nn.SineLayer(hidden_dim, hidden_dim))
        output_layers.append(nn.Linear(hidden_dim, 1))
        output_layers.append(nn.Sigmoid())
        
        self.patch_net   = nn.Sequential(*patch_layers)
        self.spatial_net = nn.Sequential(*spatial_layers)
        self.output_net  = nn.Sequential(*output_layers)
        
        if not weight_decay:
            weight_decay = [0.1, 0.0001, 0.001]
            
        self.params  = []
        self.params += [{"params": self.spatial_net.parameters(), "weight_decay": weight_decay[0]}]
        self.params += [{"params": self.patch_net.parameters(),   "weight_decay": weight_decay[1]}]
        self.params += [{"params": self.output_net.parameters(),  "weight_decay": weight_decay[2]}]
        
    def forward(self, patch: torch.Tensor, spatial: torch.Tensor, depth: torch.Tensor = None) -> torch.Tensor:
        fd    = fd_3d(patch, depth, self.alpha)
        patch = patch * fd
        for d_layer, p_layer in zip(self.depth_net, self.patch_net):
            depth = d_layer(depth)
            patch = p_layer(patch) * fd_2d(patch, depth, self.alpha)
        # patch   = self.patch_net(patch)
        spatial = self.spatial_net(spatial)
        return self.output_net(torch.cat((patch, spatial), -1))
