#!/usr/bin/env python
# -*- coding: utf-8 -*-

__all__ = [
    "DenoiseNet",
    "INF1_Patch",
    "INF1_Spatial",
    "INF2",
    "INF4",
]

import torch

from mon.core import nn


class INF1_Spatial(nn.Module):
    """Implicit Neural Function (INF) for 1-way residual reconstruction,
    i.e., f: (p) -> r.
    
    References:
        - Code: https://github.com/lly-louis/INF
        - Code: https://github.com/ctom2/colie
    """
    
    def __init__(
        self,
        s_in_features   : int   = 2,
        p_in_features   : int   = 1,
        hidden_dim      : int   = 256,
        num_layers      : int   = 4,
        add_layers      : int   = 2,
        nonlinear       : str   = "finer",
        w0              : float = 30.0,
        first_bias_scale: float = 20.0,
        reduce_channels : bool  = False,
        weight_decay    : tuple = (0.1, 0.0001, 0.001),
    ):
        super().__init__()
        # Construct MLP/INF
        spatial_layers = [nn.INRLayer(s_in_features, hidden_dim, True, nonlinear, w0, is_first=True, first_bias_scale=first_bias_scale)]
        for _ in range(1, add_layers - 2):
            spatial_layers.append(nn.INRLayer(hidden_dim, hidden_dim, True, nonlinear, w0, first_bias_scale=first_bias_scale))
        spatial_layers.append(nn.INRLayer(hidden_dim, hidden_dim, True, nonlinear, w0, first_bias_scale=first_bias_scale))
        
        output_layers = [nn.INRLayer(hidden_dim, hidden_dim, True, nonlinear, w0, first_bias_scale=first_bias_scale)]
        for _ in range(add_layers + 1, num_layers - 1):
            output_layers.append(nn.INRLayer(hidden_dim, hidden_dim, True, nonlinear, w0, first_bias_scale=first_bias_scale))
        output_layers.append(nn.INRLayer(hidden_dim, 1, True, nonlinear, w0, is_last=True, first_bias_scale=first_bias_scale))
        
        self.spatial_net = nn.Sequential(*spatial_layers)
        self.output_net  = nn.Sequential(*output_layers)

        self.params  = []
        self.params += [{"params": self.spatial_net.parameters(), "weight_decay": weight_decay[0]}]
        self.params += [{"params": self.output_net.parameters(), "weight_decay": weight_decay[2]}]
        
    def forward(self, spatial: torch.Tensor, patch: torch.Tensor) -> torch.Tensor:
        return self.output_net(self.spatial_net(spatial))


class INF1_Patch(nn.Module):
    """Implicit Neural Function (INF) for 1-way residual reconstruction,
    i.e., f: (v) -> r.
    
    References:
        - Code: https://github.com/lly-louis/INF
        - Code: https://github.com/ctom2/colie
    """
    
    def __init__(
        self,
        s_in_features   : int   = 2,
        p_in_features   : int   = 1,
        hidden_dim      : int   = 256,
        num_layers      : int   = 4,
        add_layers      : int   = 2,
        nonlinear       : str   = "finer",
        w0              : float = 30.0,
        first_bias_scale: float = 20.0,
        reduce_channels : bool  = False,
        weight_decay    : tuple = (0.1, 0.0001, 0.001),
    ):
        super().__init__()
        # Construct MLP/INF
        patch_layers = [nn.INRLayer(p_in_features, hidden_dim, True, nonlinear, w0, is_first=True, first_bias_scale=first_bias_scale)]
        for _ in range(1, add_layers - 2):
            patch_layers.append(nn.INRLayer(hidden_dim, hidden_dim, True, nonlinear, w0, first_bias_scale=first_bias_scale))
        patch_layers.append(nn.INRLayer(hidden_dim, hidden_dim, True, nonlinear, w0, first_bias_scale=first_bias_scale))
        
        output_layers = [nn.INRLayer(hidden_dim, hidden_dim, True, nonlinear, w0, first_bias_scale=first_bias_scale)]
        for _ in range(add_layers + 1, num_layers - 1):
            output_layers.append(nn.INRLayer(hidden_dim, hidden_dim, True, nonlinear, w0, first_bias_scale=first_bias_scale))
        output_layers.append(nn.INRLayer(hidden_dim, 1, True, nonlinear, w0, is_last=True, first_bias_scale=first_bias_scale))
        
        self.patch_net  = nn.Sequential(*patch_layers)
        self.output_net = nn.Sequential(*output_layers)

        self.params  = []
        self.params += [{"params": self.patch_net.parameters(),  "weight_decay": weight_decay[1]}]
        self.params += [{"params": self.output_net.parameters(), "weight_decay": weight_decay[2]}]
        
    def forward(self, spatial: torch.Tensor, patch: torch.Tensor) -> torch.Tensor:
        return self.output_net(self.patch_net(patch))


class INF2(nn.Module):
    """Implicit Neural Function (INF) for 2-way residual reconstruction,
    i.e., f: (p,v) -> r.
    
    References:
        - Code: https://github.com/lly-louis/INF
        - Code: https://github.com/ctom2/colie
    """
    
    def __init__(
        self,
        s_in_features   : int   = 2,
        p_in_features   : int   = 1,
        hidden_dim      : int   = 256,
        num_layers      : int   = 4,
        add_layers      : int   = 2,
        nonlinear       : str   = "finer",
        w0              : float = 30.0,
        first_bias_scale: float = 20.0,
        reduce_channels : bool  = False,
        weight_decay    : tuple = (0.1, 0.0001, 0.001),
    ):
        super().__init__()
        # Construct MLP/INF
        mid_features   = hidden_dim // 2 if reduce_channels else hidden_dim

        spatial_layers = [nn.INRLayer(s_in_features, hidden_dim, True, nonlinear, w0, is_first=True, first_bias_scale=first_bias_scale)]
        patch_layers   = [nn.INRLayer(p_in_features, hidden_dim, True, nonlinear, w0, is_first=True, first_bias_scale=first_bias_scale)]
        for _ in range(1, add_layers - 2):
            spatial_layers.append(nn.INRLayer(hidden_dim, hidden_dim, True, nonlinear, w0, first_bias_scale=first_bias_scale))
            patch_layers.append(  nn.INRLayer(hidden_dim, hidden_dim, True, nonlinear, w0, first_bias_scale=first_bias_scale))
        spatial_layers.append(nn.INRLayer(hidden_dim, mid_features, True, nonlinear, w0, first_bias_scale=first_bias_scale))
        patch_layers.append(  nn.INRLayer(hidden_dim, mid_features, True, nonlinear, w0, first_bias_scale=first_bias_scale))
        
        output_layers = [nn.INRLayer(mid_features * 2, hidden_dim, True, nonlinear, w0, first_bias_scale=first_bias_scale)]
        for _ in range(add_layers + 1, num_layers - 1):
            output_layers.append(nn.INRLayer(hidden_dim, hidden_dim, True, nonlinear, w0, first_bias_scale=first_bias_scale))
        output_layers.append(nn.INRLayer(hidden_dim, 1, True, nonlinear, w0, is_last=True, first_bias_scale=first_bias_scale))
        
        self.spatial_net = nn.Sequential(*spatial_layers)
        self.patch_net   = nn.Sequential(*patch_layers)
        self.output_net  = nn.Sequential(*output_layers)

        self.params  = []
        self.params += [{"params": self.spatial_net.parameters(), "weight_decay": weight_decay[0]}]
        self.params += [{"params": self.patch_net.parameters(),   "weight_decay": weight_decay[1]}]
        self.params += [{"params": self.output_net.parameters(),  "weight_decay": weight_decay[2]}]
        
    def forward(self, spatial: torch.Tensor, patch: torch.Tensor) -> torch.Tensor:
        return self.output_net(torch.cat((self.spatial_net(spatial), self.patch_net(patch)), -1))


class INF4(nn.Module):
    """Implicit Neural Function (INF) for 4-way residual reconstruction,
    i.e., f: (p,v,d,e) -> r.
    
    References:
        - Code: https://github.com/lly-louis/INF
        - Code: https://github.com/ctom2/colie
    """
    
    def __init__(
        self,
        s_in_features   : int   = 2,
        p_in_features   : int   = 1,
        hidden_dim      : int   = 256,
        num_layers      : int   = 4,
        add_layers      : int   = 2,
        w0              : float = 30.0,
        first_bias_scale: float = 20.0,
        nonlinear       : str   = "finer",
        reduce_channels : bool  = False,
        weight_decay    : tuple = (0.1, 0.0001, 0.001),
    ):
        super().__init__()
        # Construct MLP/INF
        mid_features   = hidden_dim // 4 if reduce_channels else hidden_dim

        spatial_layers = [nn.INRLayer(s_in_features, hidden_dim, True, nonlinear, w0, is_first=True, first_bias_scale=first_bias_scale)]
        patch_i_layers = [nn.INRLayer(p_in_features, hidden_dim, True, nonlinear, w0, is_first=True, first_bias_scale=first_bias_scale)]
        patch_d_layers = [nn.INRLayer(p_in_features, hidden_dim, True, nonlinear, w0, is_first=True, first_bias_scale=first_bias_scale)]
        patch_e_layers = [nn.INRLayer(p_in_features, hidden_dim, True, nonlinear, w0, is_first=True, first_bias_scale=first_bias_scale)]
        for _ in range(1, add_layers - 2):
            spatial_layers.append(nn.INRLayer(hidden_dim, hidden_dim, True, nonlinear, w0, first_bias_scale=first_bias_scale))
            patch_i_layers.append(nn.INRLayer(hidden_dim, hidden_dim, True, nonlinear, w0, first_bias_scale=first_bias_scale))
            patch_d_layers.append(nn.INRLayer(hidden_dim, hidden_dim, True, nonlinear, w0, first_bias_scale=first_bias_scale))
            patch_e_layers.append(nn.INRLayer(hidden_dim, hidden_dim, True, nonlinear, w0, first_bias_scale=first_bias_scale))
        spatial_layers.append(nn.INRLayer(hidden_dim, mid_features, True, nonlinear, w0, first_bias_scale=first_bias_scale))
        patch_i_layers.append(nn.INRLayer(hidden_dim, mid_features, True, nonlinear, w0, first_bias_scale=first_bias_scale))
        patch_d_layers.append(nn.INRLayer(hidden_dim, mid_features, True, nonlinear, w0, first_bias_scale=first_bias_scale))
        patch_e_layers.append(nn.INRLayer(hidden_dim, mid_features, True, nonlinear, w0, first_bias_scale=first_bias_scale))
        
        output_layers = [nn.INRLayer(mid_features * 4, hidden_dim, True, nonlinear, w0, first_bias_scale=first_bias_scale)]
        for _ in range(add_layers + 1, num_layers - 1):
            output_layers.append(nn.INRLayer(hidden_dim, hidden_dim, True, nonlinear, w0, first_bias_scale=first_bias_scale))
        output_layers.append(nn.INRLayer(hidden_dim, 1, True, nonlinear, w0, is_last=True, first_bias_scale=first_bias_scale))
        
        self.spatial_net = nn.Sequential(*spatial_layers)
        self.patch_i_net = nn.Sequential(*patch_i_layers)
        self.patch_d_net = nn.Sequential(*patch_d_layers)
        self.patch_e_net = nn.Sequential(*patch_e_layers)
        self.output_net  = nn.Sequential(*output_layers)

        self.params  = []
        self.params += [{"params": self.spatial_net.parameters(), "weight_decay": weight_decay[0]}]
        self.params += [{"params": self.patch_i_net.parameters(), "weight_decay": weight_decay[1]}]
        self.params += [{"params": self.patch_d_net.parameters(), "weight_decay": weight_decay[1]}]
        self.params += [{"params": self.patch_e_net.parameters(), "weight_decay": weight_decay[1]}]
        self.params += [{"params": self.output_net.parameters(),  "weight_decay": weight_decay[2]}]
        
    def forward(self, spatial: torch.Tensor, patch_i: torch.Tensor, patch_d: torch.Tensor, patch_e: torch.Tensor) -> torch.Tensor:
        output = self.output_net(
            torch.cat(
                (self.spatial_net(spatial),
                        self.patch_i_net(patch_i),
                        self.patch_d_net(patch_d),
                        self.patch_e_net(patch_e)),
                -1)
        )
        return output


class DenoiseNet(nn.Module):
    
    def __init__(self, in_channels: int, embedded_channels: int = 48):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels,       embedded_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(embedded_channels, embedded_channels, 3, padding=1)
        self.conv3 = nn.Conv2d(embedded_channels, in_channels,       1)
        self.act   = nn.LeakyReLU(negative_slope=0.2, inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.conv3(x)
        return x
