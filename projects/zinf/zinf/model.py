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
import torch

from mon.constants import MODELS
from mon.core import image as I, MLType, ModelMixin, nn, Path, Task, log
from . import loss as L
from .inr import FINER, PE_FINER, PE_SIREN, SIREN
from .utils import *

current_file = Path(__file__).absolute()
root_dir     = current_file.parents[1]
INRS         = {
    "siren"   : SIREN,
    "finer"   : FINER,
    "pe_siren": PE_SIREN,
    "pe_finer": PE_FINER,
}


@MODELS.register(name="zinf", arch="zinf")
class ZINF(nn.Module, ModelMixin):
    """ZINF model for low-light image enhancement."""
    
    arch     : str          = "zinf"
    name     : str          = "zinf"
    tasks    : list[Task]   = [Task.LLE]
    mltypes  : list[MLType] = [MLType.ZERO_SHOT]
    model_dir: Path         = root_dir
    zoo      : dict         = box.Box()
    
    def __init__(
        self,
        window_size: int   = 7,
        hidden_dim : int   = 256,
        num_layers : int   = 4,
        add_layers : int   = 2,
        inr        : str   = "siren",
        use_depth  : bool  = False,
        L          : float = 0.5,
        iters      : int   = 100,
    ):
        super().__init__()
        self.window_size = window_size
        self.hidden_dim  = hidden_dim
        self.use_depth   = use_depth
        self.L           = L
        self.iters       = iters

        self.model = INRS[inr](
            patch_dim  = self.window_size ** 2,
            hidden_dim = hidden_dim,
            num_layers = num_layers,
            add_layers = add_layers,
        )
        self.state_dict = self.model.state_dict()
        
    def forward(self, image: torch.Tensor, depth: torch.Tensor = None, save_debug: bool = False) -> torch.Tensor:
        window_size = self.window_size
        imgsz       = self.hidden_dim
        device      = image.device
        
        # Preprocess
        hvi         = I.RGBToHVI(requires_grad=False).to(device)
        image_hvi   = hvi.rgb_to_hvi(image)
        image_hv    = image_hvi[:, 0:2, :, :]
        image_i     = image_hvi[:, 2:3, :, :]
        image_i_lr  = interpolate_image(image_i, imgsz)
        patches     = create_patches(image_i_lr, window_size)
        coords      = create_coords(imgsz).to(device)
        if self.use_depth:
            depth_lr  = interpolate_image(depth, imgsz)       if depth is not None else None
            patches_d = create_patches(depth_lr, window_size) if depth is not None else None
        else:
            depth_lr  = None
            patches_d = None
        
        # Optimize
        self.model.load_state_dict(self.state_dict)
        self.model.train()
        optimizer = nn.Adam(self.model.parameters(), lr=1e-5, betas=(0.9, 0.999), weight_decay=3e-4)
        L_exp     = L.L_exp(16, self.L).to(device)
        L_tv      = L.L_tv().to(device)

        image_i_fixed_lr = None
        for i in range(self.iters):
            optimizer.zero_grad()
            illu_res_lr      = self.model(patches, coords, patches_d)
            illu_res_lr      = illu_res_lr.view(1, 1, imgsz, imgsz)
            illu_lr          = illu_res_lr + image_i_lr
            image_i_fixed_lr = image_i_lr / (illu_lr + 1e-4)
            
            l_spa      = torch.mean(torch.abs(torch.pow(illu_lr - image_i_lr, 2)))
            l_tv       = L_tv(illu_lr)
            l_exp      = torch.mean(L_exp(illu_lr))
            l_sparsity = torch.mean(image_i_fixed_lr)
            loss       = l_spa + 20 * l_tv + 8 * l_exp + 5 * l_sparsity
            
            loss.backward()
            optimizer.step()
        
        # Postprocess
        image_i_fixed   = filter_up(image_i_lr, image_i_fixed_lr, image_i)
        image_hsv_fixed = torch.cat((image_hv, image_i_fixed), dim=1).to(device)
        image_rgb_fixed = hvi.hvi_to_rgb(image_hsv_fixed)
        enhanced        = image_rgb_fixed

        return {
            "enhanced": enhanced,
        }
