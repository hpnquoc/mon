#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements the paper: "Zero-Shot Implicit Neural Fusion Network for Multimodal
Low-Light Image Enhancement".
"""

__all__ = [
    "ZINF",
]

from typing import Literal

import kornia
import numpy as np
import torch

from mon import core, nn
from mon.constants import MLType, MODELS, Task
from mon.nn import _size_2_t, functional as F
from mon.vision import filtering, types
from mon.vision.enhance import base

current_file = core.Path(__file__).absolute()
current_dir  = current_file.parents[0]
INR_AF       = nn.inr_layer.INR_AF
MAPPING_FUNC = Literal["p", "b", "d", "e", "pb", "pd", "pe", "pbde"]


# ----- Loss -----
class Loss(nn.Loss):
    
    def __init__(
        self,
        loss_e_mean: float = 0.1,
        loss_w_f   : float = 1.0,
        loss_w_s   : float = 5.0,
        loss_w_e   : float = 8.0,
        loss_w_tv  : float = 20.0,
        loss_w_de  : float = 10.0,
        reduction  : Literal["none", "mean", "sum"] = "mean",
        verbose    : bool  = False,
    ):
        super().__init__(reduction=reduction)
        self.loss_w_f   = loss_w_f
        self.loss_w_s   = loss_w_s
        self.loss_w_e   = loss_w_e
        self.loss_w_tv  = loss_w_tv
        self.loss_w_de  = loss_w_de
        self.verbose    = verbose
        
        self.loss_e     = nn.ExposureValueControlLoss(16, loss_e_mean, reduction=reduction)
        self.loss_tv    = nn.TotalVariationLoss(reduction=reduction)
        self.loss_depth = nn.DepthAwareIlluminationLoss(reduction=reduction)

    def forward(
        self,
        illu_lr         : torch.Tensor,
        image_i_lr      : torch.Tensor,
        image_i_fixed_lr: torch.Tensor,
        depth_lr        : torch.Tensor = None,
    ) -> torch.Tensor:
        loss_f  = self.loss_w_f  * torch.mean(torch.abs(torch.pow(illu_lr - image_i_lr, 2)))
        loss_s  = self.loss_w_s  * torch.mean(image_i_fixed_lr)
        loss_e  = self.loss_w_e  * torch.mean(self.loss_e(illu_lr))
        loss_tv = self.loss_w_tv * self.loss_tv(illu_lr)
        loss_de = 0.0
        if depth_lr is not None:
            loss_de = self.loss_depth(illu_lr, depth_lr)
        loss_de = self.loss_w_de * loss_de
        loss    = loss_f + loss_s + loss_e + loss_tv + loss_de # + loss_c
        
        if self.verbose:
            core.console.log(
                f"loss_f : {loss_f:.4f}, "
                f"loss_s : {loss_s:.4f}, "
                f"loss_e : {loss_e:.4f}, "
                f"loss_tv: {loss_tv:.4f}, "
                f"loss_de: {loss_de:.4f}, "
            )
        
        return loss


# ----- Module -----
class INF1_Spatial(nn.Module):
    """Implicit Neural Function (INF) for 1-way residual reconstruction,
    i.e., f: (p) -> r.
    
    References:
        - https://github.com/lly-louis/INF
        - https://github.com/ctom2/colie
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
        s_nonlinear     : str   = "finer",
        nonlinear       : str   = "finer",
        reduce_channels : bool  = False,
        weight_decay            = [0.1, 0.0001, 0.001],
    ):
        super().__init__()
        # Construct MLP/INF
        spatial_layers = [nn.INRLayer(s_in_features, hidden_dim, s_nonlinear, w0=w0, first_bias_scale=first_bias_scale, is_first=True)]
        for _ in range(1, add_layers - 2):
            spatial_layers.append(nn.INRLayer(hidden_dim, hidden_dim, s_nonlinear, w0=w0, first_bias_scale=first_bias_scale))
        spatial_layers.append(nn.INRLayer(hidden_dim, hidden_dim, s_nonlinear, w0=w0, first_bias_scale=first_bias_scale))
        
        output_layers = [nn.INRLayer(hidden_dim, hidden_dim, nonlinear, w0=w0, first_bias_scale=first_bias_scale)]
        for _ in range(add_layers + 1, num_layers - 1):
            output_layers.append(nn.INRLayer(hidden_dim, hidden_dim, nonlinear, w0=w0, first_bias_scale=first_bias_scale))
        output_layers.append(nn.INRLayer(hidden_dim, 1, nonlinear, w0=w0, first_bias_scale=first_bias_scale, is_last=True))
        
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
        - https://github.com/lly-louis/INF
        - https://github.com/ctom2/colie
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
        weight_decay            = [0.1, 0.0001, 0.001],
    ):
        super().__init__()
        # Construct MLP/INF
        patch_layers = [nn.INRLayer(p_in_features, hidden_dim, nonlinear, w0=w0, first_bias_scale=first_bias_scale, is_first=True)]
        for _ in range(1, add_layers - 2):
            patch_layers.append(nn.INRLayer(hidden_dim, hidden_dim, nonlinear, w0=w0, first_bias_scale=first_bias_scale))
        patch_layers.append(nn.INRLayer(hidden_dim, hidden_dim, nonlinear, w0=w0, first_bias_scale=first_bias_scale))
        
        output_layers = [nn.INRLayer(hidden_dim, hidden_dim, nonlinear, w0=w0, first_bias_scale=first_bias_scale)]
        for _ in range(add_layers + 1, num_layers - 1):
            output_layers.append(nn.INRLayer(hidden_dim, hidden_dim, nonlinear, w0=w0, first_bias_scale=first_bias_scale))
        output_layers.append(nn.INRLayer(hidden_dim, 1, nonlinear, w0=w0, first_bias_scale=first_bias_scale, is_last=True))
        
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
        - https://github.com/lly-louis/INF
        - https://github.com/ctom2/colie
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
        weight_decay            = [0.1, 0.0001, 0.001],
    ):
        super().__init__()
        # Construct MLP/INF
        mid_features   = hidden_dim // 2 if reduce_channels else hidden_dim

        spatial_layers = [nn.INRLayer(s_in_features, hidden_dim, nonlinear, w0=w0, first_bias_scale=first_bias_scale, is_first=True)]
        patch_layers   = [nn.INRLayer(p_in_features, hidden_dim, nonlinear, w0=w0, first_bias_scale=first_bias_scale, is_first=True)]
        for _ in range(1, add_layers - 2):
            spatial_layers.append(nn.INRLayer(hidden_dim, hidden_dim, nonlinear, w0=w0, first_bias_scale=first_bias_scale))
            patch_layers.append(  nn.INRLayer(hidden_dim, hidden_dim, nonlinear, w0=w0, first_bias_scale=first_bias_scale))
        spatial_layers.append(nn.INRLayer(hidden_dim, mid_features, nonlinear, w0=w0, first_bias_scale=first_bias_scale))
        patch_layers.append(  nn.INRLayer(hidden_dim, mid_features, nonlinear, w0=w0, first_bias_scale=first_bias_scale))
        
        output_layers = [nn.INRLayer(mid_features * 2, hidden_dim, nonlinear, w0=w0, first_bias_scale=first_bias_scale)]
        for _ in range(add_layers + 1, num_layers - 1):
            output_layers.append(nn.INRLayer(hidden_dim, hidden_dim, nonlinear, w0=w0, first_bias_scale=first_bias_scale))
        output_layers.append(nn.INRLayer(hidden_dim, 1, nonlinear, w0=w0, first_bias_scale=first_bias_scale, is_last=True))
        
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
        - https://github.com/lly-louis/INF
        - https://github.com/ctom2/colie
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
        weight_decay            = [0.1, 0.0001, 0.001],
    ):
        super().__init__()
        # Construct MLP/INF
        mid_features   = hidden_dim // 4 if reduce_channels else hidden_dim

        spatial_layers = [nn.INRLayer(s_in_features, hidden_dim, nonlinear, w0=w0, first_bias_scale=first_bias_scale, is_first=True)]
        patch_i_layers = [nn.INRLayer(p_in_features, hidden_dim, nonlinear, w0=w0, first_bias_scale=first_bias_scale, is_first=True)]
        patch_d_layers = [nn.INRLayer(p_in_features, hidden_dim, nonlinear, w0=w0, first_bias_scale=first_bias_scale, is_first=True)]
        patch_e_layers = [nn.INRLayer(p_in_features, hidden_dim, nonlinear, w0=w0, first_bias_scale=first_bias_scale, is_first=True)]
        for _ in range(1, add_layers - 2):
            spatial_layers.append(nn.INRLayer(hidden_dim, hidden_dim, nonlinear, w0=w0, first_bias_scale=first_bias_scale))
            patch_i_layers.append(nn.INRLayer(hidden_dim, hidden_dim, nonlinear, w0=w0, first_bias_scale=first_bias_scale))
            patch_d_layers.append(nn.INRLayer(hidden_dim, hidden_dim, nonlinear, w0=w0, first_bias_scale=first_bias_scale))
            patch_e_layers.append(nn.INRLayer(hidden_dim, hidden_dim, nonlinear, w0=w0, first_bias_scale=first_bias_scale))
        spatial_layers.append(nn.INRLayer(hidden_dim, mid_features, nonlinear, w0=w0, first_bias_scale=first_bias_scale))
        patch_i_layers.append(nn.INRLayer(hidden_dim, mid_features, nonlinear, w0=w0, first_bias_scale=first_bias_scale))
        patch_d_layers.append(nn.INRLayer(hidden_dim, mid_features, nonlinear, w0=w0, first_bias_scale=first_bias_scale))
        patch_e_layers.append(nn.INRLayer(hidden_dim, mid_features, nonlinear, w0=w0, first_bias_scale=first_bias_scale))
        
        output_layers = [nn.INRLayer(mid_features * 4, hidden_dim, nonlinear, w0=w0, first_bias_scale=first_bias_scale)]
        for _ in range(add_layers + 1, num_layers - 1):
            output_layers.append(nn.INRLayer(hidden_dim, hidden_dim, nonlinear, w0=w0, first_bias_scale=first_bias_scale))
        output_layers.append(nn.INRLayer(hidden_dim, 1, nonlinear, w0=w0, first_bias_scale=first_bias_scale, is_last=True))
        
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
class ZINF(base.ImageEnhancementModel):
    """ZINF model for low-light image enhancement."""

    arch     : str          = "zinf"
    name     : str          = "zinf"
    tasks    : list[Task]   = [Task.LLE]
    mltypes  : list[MLType] = [MLType.ZERO_SHOT]
    model_dir: core.Path    = current_dir
    zoo      : dict         = {}
    
    def __init__(
        self,
        mapping_func     : str   = "pbde",
        color_space      : str   = "hvi",
        window_size      : int   = 9,
        down_size        : int   = 256,
        num_layers       : int   = 4,
        add_layers       : int   = 2,
        w0               : float = 30.0,
        first_bias_scale : float = 20.0,
        use_ff           : bool  = True,
        ff_gaussian_scale: float = 10.0,
        nonlinear        : str   = "finer",
        reduce_channels  : bool  = False,
        depth_threshold  : float = 1.0,
        edge_threshold   : float = 0.05,
        # Post-process
        use_denoise      : bool  = True,
        denoise_ksize    : int   = 7,
        denoise_color    : float = 0.1,
        denoise_space    : float = 1.5,
        iters            : int   = 100,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.mapping_func    = mapping_func
        self.color_space     = color_space
        self.window_size     = window_size
        self.down_size       = down_size
        self.depth_threshold = depth_threshold
        self.edge_threshold  = edge_threshold
        self.use_denoise     = use_denoise
        self.denoise_ksize   = (denoise_ksize, denoise_ksize)
        self.denoise_color   = denoise_color
        self.denoise_space   = (denoise_space, denoise_space)
        self.iters           = iters

        # Model
        patch_dim  = self.window_size ** 2
        hidden_dim = self.down_size
        if use_ff:
            self.register_buffer("B1", torch.randn((hidden_dim, 2))         * ff_gaussian_scale)
            self.register_buffer("B2", torch.randn((hidden_dim, patch_dim)) * ff_gaussian_scale)
            s_in_features = hidden_dim * 2
            p_in_features = hidden_dim * 2
        else:
            self.B1       = None
            self.B2       = None
            s_in_features = 2
            p_in_features = patch_dim

        if self.mapping_func in ["p"]:
            inf = INF1_Spatial
        elif self.mapping_func in ["b", "e", "d"]:
            inf = INF1_Patch
        elif self.mapping_func in ["pb", "pd", "pe"]:
            inf = INF2
        elif self.mapping_func in ["pbde"]:
            inf = INF4
        else:
            raise ValueError(f"[mapping_func] must be one of {MAPPING_FUNC}, got {self.mapping_func}.")
        self.inf = inf(
            s_in_features    = s_in_features,
            p_in_features    = p_in_features,
            hidden_dim       = hidden_dim,
            num_layers       = num_layers,
            add_layers       = add_layers,
            w0               = w0,
            first_bias_scale = first_bias_scale,
            nonlinear        = nonlinear,
            reduce_channels  = reduce_channels,
            weight_decay     = [0.1, 0.0001, 0.001],
        )
        self.gf = filtering.FastGuidedFilter(radius=3)
        self.bf = kornia.filters.BilateralBlur(self.denoise_ksize, self.denoise_color, self.denoise_space)

        # Optimizer
        self.configure_optimizers()
        
        # Loss
        self.loss = Loss(
            loss_e_mean = self.loss["loss_e_mean"],
            loss_w_f    = self.loss["loss_w_f"],
            loss_w_s    = self.loss["loss_w_s"],
            loss_w_e    = self.loss["loss_w_e"],
            loss_w_tv   = self.loss["loss_w_tv"],
            loss_w_de   = self.loss["loss_w_de"],
            verbose     = self.loss["verbose"],
        )
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)
        self.initial_state_dict = self.state_dict()
    
    # ----- Initialize -----
    def init_weights(self, m: nn.Module):
        """Initializes the model's weights.
    
        Args:
            m: ``nn.Module`` to initialize weights for.
        """
        pass
    
    def compute_efficiency_score(self, imgsz: _size_2_t = 512) -> tuple[float, float]:
        """Compute model efficiency score (FLOPs, params).

        Args:
            imgsz: Input size as ``int`` or [H, W]. Default is ``512``.

        Returns:
            Tuple of (FLOPs, parameter count) as ``float`` values.
        """
        from fvcore.nn import parameter_count
        
        h, w      = types.image_size(imgsz)
        image     = torch.rand(1, 1, h, w).to(self.device)
        image_lr  = self.interpolate_image(image, self.down_size)
        #
        spatial   = self.create_coords(self.down_size).to(self.device)
        patch_i   = self.create_patches(image_lr, self.window_size)
        #
        spatial_ff = self.ff_embedding(spatial, self.B1)
        patch_i_ff = self.ff_embedding(patch_i, self.B2)
        #
        datapoint = {
            "image_i_lr": image_lr,
            "depth_lr"  : image_lr,
            "spatial"   : spatial_ff,
            "patch_i"   : patch_i_ff,
            "patch_d"   : patch_i_ff,
            "patch_e"   : patch_i_ff,
        }

        flops, params = core.thop.custom_profile(self, inputs=datapoint, verbose=False)
        params        = self.params                if hasattr(self, "params") and params == 0 else params
        params        = parameter_count(self)      if hasattr(self, "params")  else params
        params        = sum(list(params.values())) if isinstance(params, dict) else params

        return flops, params
    
    # ----- Forward Pass -----
    def forward_loss(self, datapoint: dict, *args, **kwargs) -> dict:
        """Computes forward pass and loss.
    
        Args:
            datapoint: ``dict`` with datapoint attributes.
    
        Returns:
            ``dict`` of predictions with ``"loss"`` and ``"enhanced"`` keys.
        """
        # Forward
        outputs  = self.forward(datapoint=datapoint, *args, **kwargs)
        
        # Loss
        depth_lr         = datapoint["depth_lr"]
        illu_lr          = outputs["illu_lr"]
        image_i_lr       = outputs["image_i_lr"]
        image_i_fixed_lr = outputs["image_i_fixed_lr"]
        loss             = self.loss(illu_lr, image_i_lr, image_i_fixed_lr, depth_lr)

        return outputs | {
            "loss": loss,
        }
        
    def forward(self, datapoint: dict, *args, **kwargs) -> dict:
        """Performs forward pass of the model.
    
        Args:
            datapoint: ``dict`` with datapoint attributes.
    
        Returns:
            ``dict`` of predictions with ``"enhanced"`` keys.
        """
        # Input
        image_i_lr = datapoint["image_i_lr"]
        depth_lr   = datapoint["depth_lr"]
        spatial    = datapoint["spatial"]
        patch_i    = datapoint["patch_i"]
        patch_d    = datapoint["patch_d"]
        patch_e    = datapoint["patch_e"]

        # Mapping
        if self.mapping_func in ["p", "b", "pb"]:
            illu_res_lr = self.inf(spatial, patch_i)
        elif self.mapping_func in ["d", "pd"]:
            illu_res_lr = self.inf(spatial, patch_d)
        elif self.mapping_func in ["e", "pe"]:
            illu_res_lr = self.inf(spatial, patch_e)
        elif self.mapping_func in ["pbde"]:
            illu_res_lr = self.inf(spatial, patch_i, patch_d, patch_e)
        else:
            raise ValueError(f"[mapping_func] must be one of {MAPPING_FUNC}, got {self.mapping_func}.")
        illu_res_lr = illu_res_lr.view(1, 1, self.down_size, self.down_size)
        
        # Enhance
        if self.depth_threshold > 0:
            illu_res_lr = illu_res_lr * (1 + self.depth_threshold * (1 - depth_lr / depth_lr.max()))
        illu_lr          = image_i_lr + illu_res_lr
        image_i_fixed_lr = image_i_lr / (illu_lr + 1e-8)

        return {
            "illu_res_lr"     : illu_res_lr,
            "illu_lr"         : illu_lr,
            "image_i_lr"      : image_i_lr,
            "image_i_fixed_lr": image_i_fixed_lr,
        }

    # ----- Predict -----
    def infer(
        self,
        datapoint: dict,
        reset    : bool              = True,
        timers   : core.TimeProfiler = None,
        *args, **kwargs
    ) -> dict:
        """Infers model output with optional processing.
    
        Args:
            datapoint: ``dict`` with datapoint attributes.
            reset: Whether to reset the weights before training. Default is ``True``.
            timers: ``TimeProfiler`` for measuring time.

        Returns:
            ``dict`` of model predictions.
    
        Notes:
            Override for custom pre/post-processing; defaults to ``self.forward()``.
        """
        # Initialize training components
        if reset:
            self.load_state_dict(self.initial_state_dict, strict=False)
        optimizer = self.optimizer.get("optimizer", None)
        optimizer = optimizer or nn.Adam(self.parameters(), lr=0.00001, weight_decay=0.0003)

        # Preprocess
        timers.preprocess.tick() if timers is not None else None
        hvi        = types.RGBToHVI(requires_grad=True).to(self.device)
        image_rgb  = datapoint["image"].to(self.device)
        image_hvi  = hvi.rgb_to_hvi(image_rgb)
        image_hv   = image_hvi[:, 0:2, :, :]
        image_i    = image_hvi[:, 2:3, :, :]
        depth      = datapoint.get("depth", None)
        depth      = depth.to(self.device) if depth is not None else None
        edge       = types.boundary_aware_prior(depth, self.edge_threshold) if depth is not None else None
        #
        image_i_lr = self.interpolate_image(image_i, self.down_size)
        depth_lr   = self.interpolate_image(depth,   self.down_size) if depth is not None else None
        edge_lr    = self.interpolate_image(edge,    self.down_size) if edge  is not None else None
        #
        spatial    = self.create_coords(self.down_size).to(self.device)
        patch_i    = self.create_patches(image_i_lr, self.window_size)
        patch_d    = self.create_patches(depth_lr,   self.window_size)
        patch_e    = self.create_patches(edge_lr,    self.window_size)
        #
        spatial_ff = self.ff_embedding(spatial, self.B1)
        patch_i_ff = self.ff_embedding(patch_i, self.B2)
        patch_d_ff = self.ff_embedding(patch_d, self.B2)
        patch_e_ff = self.ff_embedding(patch_e, self.B2)
        datapoint |= {
            "image_i"   : image_i,
            "depth"     : depth,
            "edge"      : edge,
            "image_i_lr": image_i_lr,
            "depth_lr"  : depth_lr,
            "edge_lr"   : edge_lr,
            "spatial"   : spatial_ff,
            "patch_i"   : patch_i_ff,
            "patch_d"   : patch_d_ff,
            "patch_e"   : patch_e_ff,
        }
        for k, v in datapoint.items():
            if isinstance(v, torch.Tensor):
                datapoint[k] = v.to(self.device)
        timers.preprocess.tock() if timers is not None else None

        # Optimize
        timers.infer.tick() if timers is not None else None
        self.train()
        for _ in range(self.iters):
            outputs = self.forward_loss(datapoint=datapoint)
            optimizer.zero_grad()
            loss = outputs["loss"]
            loss.backward(retain_graph=True)
            optimizer.step()
        self.eval()
        outputs = self.forward(datapoint=datapoint)
        timers.infer.tock() if timers is not None else None

        # Postprocess
        timers.postprocess.tick() if timers is not None else None
        image_i_fixed_lr = outputs["image_i_fixed_lr"]
        if self.use_denoise:
            image_i_fixed_lr = self.bf(image_i_fixed_lr)
        image_i_fixed   = self.filter_up(image_i_lr, image_i_fixed_lr, image_i)
        image_hvi_fixed = torch.cat((image_hv, image_i_fixed), dim=1)
        image_rgb_fixed = hvi.hvi_to_rgb(image_hvi_fixed)
        if self.use_denoise:
            image_rgb_fixed = self.bf(image_rgb_fixed)
        timers.postprocess.tock() if timers is not None else None

        # Return
        return outputs | {
            "image_hvi"      : image_hvi,
            "image_hvi_fixed": image_hvi_fixed,
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
        }

    # ----- Utils -----
    def create_coords(self, down_size: int) -> torch.Tensor:
        """Creates a coordinates grid.

        Args:
            down_size: The size of the coordinates grid.
        """
        h, w   = down_size, down_size
        coords = np.dstack(np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w)))
        coords = torch.from_numpy(coords).float()
        return coords

    def create_patches(self, image: torch.Tensor, kernel_size: int = 1) -> torch.Tensor:
        """Creates a tensor where the channel contains patch information."""
        num_channels = types.image_num_channels(image)
        kernel       = torch.zeros((kernel_size ** 2, num_channels, kernel_size, kernel_size)).to(image.device)
        for i in range(kernel_size):
            for j in range(kernel_size):
                # kernel[int(torch.sum(kernel).item()), 0, i, j] = 1
                kernel[i + j * kernel_size, 0, i, j] = 1

        pad       = nn.ReflectionPad2d(kernel_size // 2)
        im_padded = pad(image)
        extracted = F.conv2d(im_padded, kernel, padding=0).squeeze(0)
        return torch.movedim(extracted, 0, -1)

    def interpolate_image(self, image: torch.Tensor, down_size: int) -> torch.Tensor:
        """Reshapes the image based on new resolution."""
        # return F.interpolate(image, size=(down_size, down_size), mode="bicubic")
        return F.interpolate(image, size=(down_size, down_size), mode="area")

    def ff_embedding(self, p: torch.Tensor, B: torch.Tensor = None) -> torch.Tensor:
        if B is None:
            return p
        else:
            x_proj    = (2. * np.pi * p) @ B.T
            embedding = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], axis=-1)
            return embedding

    def filter_up(self, x_lr: torch.Tensor, y_lr: torch.Tensor, x_hr: torch.Tensor) -> torch.Tensor:
        """Applies the guided filter to upscale the predicted image. """
        # gf   = filtering.FastGuidedFilter(radius=radius)
        y_hr = self.gf(x_lr, y_lr, x_hr)
        y_hr = torch.clip(y_hr, 0.0, 1.0)
        return y_hr
