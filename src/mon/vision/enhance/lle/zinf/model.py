#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements ZINF model for low-light image enhancement.

References:
    - Paper: "Zero-Shot Implicit Neural Fusion Network for Multimodal Low-Light
      Image Enhancement," arXiv 2025.
    - Code: https://github.com/phlong3105/mon
"""

__all__ = [
    "ZINF",
]

import box
import kornia
import numpy as np
import torch

from mon.constants import MODELS
from mon.core import MLType, ModelMixin, nn, Path, Task
from mon.core.dtypes.image import (
    boundary_aware_prior,
    FastGuidedFilter,
    RGBToHVI,
)
from mon.core.nn import functional as F
from . import loss as L

current_file = Path(__file__).absolute()
current_dir  = current_file.parents[0]


# ----- Module -----
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


# ----- Model -----
@MODELS.register(name="zinf", arch="zinf")
class ZINF(nn.Module, ModelMixin):
    """ZINF model for low-light image enhancement."""
    
    MAPPING_FUNC = {
        "p"   : INF1_Spatial,
        "b"   : INF1_Patch,
        "pb"  : INF2,
        "pbde": INF4,
    }
    
    arch     : str          = "zinf"
    name     : str          = "zinf"
    tasks    : list[Task]   = [Task.LLE]
    mltypes  : list[MLType] = [MLType.ZERO_SHOT]
    model_dir: Path         = current_dir
    zoo      : dict         = box.Box()
    
    def __init__(
        self,
        mapping_func   : str   = "pbde",
        window_size    : int   = 9,
        hidden_dim     : int   = 256,
        num_layers     : int   = 4,
        add_layers     : int   = 2,
        use_ff         : bool  = True,
        nonlinear      : str   = "finer",
        depth_threshold: float = 1.0,
        edge_threshold : float = 0.05,
        use_denoise    : bool  = True,
        denoise_ksize  : int   = 7,
        L              : float = 0.1,
        iters          : int   = 100,
    ):
        super().__init__()
        self.mapping_func    = mapping_func
        self.window_size     = window_size
        self.hidden_dim      = hidden_dim
        self.depth_threshold = depth_threshold
        self.edge_threshold  = edge_threshold
        self.use_denoise     = use_denoise
        self.L               = L
        self.iters           = iters
        
        if use_ff:
            self.register_buffer("B1", torch.randn((hidden_dim, 2)) * 20.0)
            self.register_buffer("B2", torch.randn((hidden_dim, self.window_size ** 2)) * 20.0)
            s_in_features = hidden_dim * 2
            p_in_features = hidden_dim * 2
        else:
            self.B1       = None
            self.B2       = None
            s_in_features = 2
            p_in_features = self.window_size ** 2
        
        if self.mapping_func not in self.MAPPING_FUNC:
            raise ValueError(f"[mapping_func] must be one of {self.MAPPING_FUNC}, got {self.mapping_func}.")
        inf = self.MAPPING_FUNC[self.mapping_func]
        self.model = inf(
            s_in_features    = s_in_features,
            p_in_features    = p_in_features,
            hidden_dim       = hidden_dim,
            num_layers       = num_layers,
            add_layers       = add_layers,
            w0               = 30.0,
            first_bias_scale = 20.0,
            nonlinear        = nonlinear,
            reduce_channels  = False,
        )
        self.state_dict = self.model.state_dict()
        
        self.gf = FastGuidedFilter(kernel_size=7)
        self.bf = kornia.filters.BilateralBlur((denoise_ksize, denoise_ksize), 0.1, (1.5, 1.5))
    
    def forward(self, image: torch.Tensor, depth: torch.Tensor = None, save_debug: bool = False) -> torch.Tensor:
        window_size = self.window_size
        imgsz       = self.hidden_dim
        device      = image.device
        
        # Preprocess
        hvi         = RGBToHVI(requires_grad=False).to(device)
        image_hvi   = hvi.rgb_to_hvi(image)
        image_hv    = image_hvi[:, 0:2, :, :]
        image_h     = image_hvi[:, 0:1, :, :]
        image_v     = image_hvi[:, 1:2, :, :]
        image_i     = image_hvi[:, 2:3, :, :]
        depth       = depth.to(device) if depth is not None else None
        edge        = boundary_aware_prior(depth, self.edge_threshold) if depth is not None else None
        #           
        image_i_lr  = self.interpolate_image(image_i, imgsz)
        depth_lr    = self.interpolate_image(depth,   imgsz) if depth is not None else None
        edge_lr     = self.interpolate_image(edge,    imgsz) if edge  is not None else None
        #           
        spatial     = self.create_coords(imgsz).to(device)
        patch_i     = self.create_patches(image_i_lr, window_size)
        patch_d     = self.create_patches(depth_lr,   window_size)
        patch_e     = self.create_patches(edge_lr,    window_size)
        #           
        spatial_ff  = self.ff_embedding(spatial, self.B1)
        patch_i_ff  = self.ff_embedding(patch_i, self.B2)
        patch_d_ff  = self.ff_embedding(patch_d, self.B2)
        patch_e_ff  = self.ff_embedding(patch_e, self.B2)
        
        # Optimize
        self.model.load_state_dict(self.state_dict)
        self.model.train()
        optimizer = nn.Adam(self.model.parameters(), lr=1e-5, betas=(0.9, 0.999), weight_decay=3e-4)
        Loss      = L.Loss(self.L).to(device)
        
        best_iter       = 0
        best_loss       = 9999
        best_state_dict = self.model.state_dict()
        for i in range(self.iters):
            optimizer.zero_grad()
            illu_res_lr = self.model(spatial_ff, patch_i_ff, patch_d_ff, patch_e_ff)
            illu_res_lr = illu_res_lr.view(1, 1, imgsz, imgsz)
            if self.depth_threshold > 0:
                illu_res_lr = illu_res_lr * (1 + self.depth_threshold * (1 - depth_lr / depth_lr.max()))
            illu_lr          = image_i_lr + illu_res_lr
            image_i_fixed_lr = image_i_lr / (illu_lr + 1e-8)
            
            loss = Loss(illu_lr, image_i_lr, image_i_fixed_lr, depth_lr)
            if loss < best_loss:
                best_iter       = i
                best_loss       = loss.item()
                best_state_dict = self.model.state_dict()
            
            loss.backward()
            optimizer.step()
        
        # Postprocess
        self.model.load_state_dict(best_state_dict, strict=False)
        self.model.eval()
        illu_res_lr = self.model(spatial_ff, patch_i_ff, patch_d_ff, patch_e_ff)
        illu_res_lr = illu_res_lr.view(1, 1, imgsz, imgsz)
        if self.depth_threshold > 0:
            illu_res_lr = illu_res_lr * (1 + self.depth_threshold * (1 - depth_lr / depth_lr.max()))
        illu_lr          = image_i_lr + illu_res_lr
        image_i_fixed_lr = image_i_lr / (illu_lr + 1e-8)
        if self.use_denoise:
            image_i_fixed_lr = self.bf(image_i_fixed_lr)
        image_i_fixed    = self.filter_up(image_i_lr, image_i_fixed_lr, image_i)
        image_hvi_fixed  = torch.cat((image_hv, image_i_fixed), dim=1)
        image_rgb_fixed  = hvi.hvi_to_rgb(image_hvi_fixed)
        if self.use_denoise:
            image_rgb_fixed = self.bf(image_rgb_fixed)
        
        enhanced = image_rgb_fixed
        outputs  = {
            "enhanced": enhanced,
        }
        if save_debug:
            outputs = {
                "image_hvi"      : image_hvi,
                "image_hvi_fixed": image_hvi_fixed,
                "image_h"        : image_h,
                "image_v"        : image_v,
                "image_i"        : image_i,
                "image_i_fixed"  : image_i_fixed,
                "depth"          : depth,
                "depth_lr"       : depth_lr,
                "edge"           : edge,
                "edge_lr"        : edge_lr,
                "spatial"        : spatial,
                "patch_i"        : patch_i,
                "patch_d"        : patch_d,
                "patch_e"        : patch_e,
                "enhanced"       : image_rgb_fixed,
            } | outputs
        return outputs
        
    # ----- Utils -----
    def create_coords(self, size: int) -> torch.Tensor:
        """Creates a coordinates grid.

        Args:
            size: The size of the coordinates grid.
        """
        h, w   = size, size
        coords = np.dstack(np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w)))
        coords = torch.from_numpy(coords).float()
        return coords

    def create_patches(self, image: torch.Tensor, kernel_size: int = 1) -> torch.Tensor:
        """Creates a tensor where the channel contains patch information."""
        b, c, h, w = image.shape
        kernel     = torch.zeros((kernel_size ** 2, c, kernel_size, kernel_size)).to(image.device)
        for i in range(kernel_size):
            for j in range(kernel_size):
                # kernel[int(torch.sum(kernel).item()), 0, i, j] = 1
                kernel[i + j * kernel_size, 0, i, j] = 1

        pad       = nn.ReflectionPad2d(kernel_size // 2)
        im_padded = pad(image)
        extracted = F.conv2d(im_padded, kernel, padding=0).squeeze(0)
        return torch.movedim(extracted, 0, -1)

    def interpolate_image(self, image: torch.Tensor, size: int) -> torch.Tensor:
        """Reshapes the image based on new resolution."""
        # return F.interpolate(image, size=(down_size, down_size), mode="bicubic")
        return F.interpolate(image, size=(size, size), mode="area")

    def ff_embedding(self, p: torch.Tensor, B: torch.Tensor = None) -> torch.Tensor:
        if B is None:
            return p
        else:
            x_proj    = (2. * np.pi * p) @ B.T
            embedding = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], axis=-1)
            return embedding

    def filter_up(self, x_lr: torch.Tensor, y_lr: torch.Tensor, x_hr: torch.Tensor) -> torch.Tensor:
        """Applies the guided filter to upscale the predicted image. """
        y_hr = self.gf(x_lr, y_lr, x_hr)
        y_hr = torch.clip(y_hr, 0.0, 1.0)
        return y_hr
