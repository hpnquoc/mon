#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements GCE-Net model for low-light image enhancement."""

__all__ = [
    "GCENet",
    "GCENet_Baseline",
    "GCENet_Ghost",
    "GCENet_MobileOne",
    "reparameterize_model",
]

import copy
from typing import Any

import box
import kornia.filters
import torch

from mon.constants import MODELS
from mon.core import image as I, MLType, ModelMixin, nn, Path, Task
from mon.core.nn import functional as F

current_file = Path(__file__).absolute()
current_dir  = current_file.parents[0]


# ----- Utils -----
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def reparameterize_model(model: nn.Module) -> nn.Module:
    """Method returns a model where a multi-branched structure used in training
    is re-parameterized into a single branch for inference.

    Args:
        model: Model to re-parameterize.
    
    Returns:
        Re-parameterized model.
    """
    # Avoid editing original graph
    model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, "reparameterize"):
            module.reparameterize()
    return model


# ----- Modules -----
class DSConv(nn.Module):
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.depth_conv = nn.Conv2d(
            in_channels  = in_channels,
            out_channels = in_channels,
            kernel_size  = 3,
            stride       = 1,
            padding      = 1,
            groups       = in_channels
        )
        self.point_conv = nn.Conv2d(
            in_channels  = in_channels,
            out_channels = out_channels,
            kernel_size  = 1,
            stride       = 1,
            padding      = 0,
            groups       = 1
        )
        self.apply(weights_init)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.depth_conv(x)
        y = self.point_conv(y)
        return y


class ConvBlock(nn.Module):
    
    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        norm        : nn.Module = nn.AdaptiveBatchNorm2d,
        use_se      : bool      = True,
    ):
        super().__init__()
        self.conv = DSConv(in_channels, out_channels)
        if norm:
            self.norm = norm(out_channels)
        else:
            self.norm = nn.Identity()
        if use_se:
            self.se = nn.SEBlock(out_channels)
        else:
            self.se = nn.Identity()
       
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.se(self.norm(self.conv(input)))


class MobileOneConv(nn.Module):
    
    def __init__(
        self,
        in_channels      : int,
        out_channels     : int,
        inference        : bool = False,
        use_se           : bool = False,
        use_act          : bool = True,
        num_conv_branches: int  = 1,
    ):
        super().__init__()
        self.depth_conv = nn.MobileOneBlock(
            in_channels       = in_channels,
            out_channels      = in_channels,
            kernel_size       = 3,
            stride            = 1,
            padding           = 1,
            groups            = in_channels,
            inference         = inference,
            use_se            = use_se,
            use_act           = use_act,
            num_conv_branches = num_conv_branches,
        )
        self.point_conv = nn.MobileOneBlock(
            in_channels       = in_channels,
            out_channels      = out_channels,
            kernel_size       = 1,
            stride            = 1,
            padding           = 0,
            groups            = 1,
            inference         = inference,
            use_se            = use_se,
            use_act           = use_act,
            num_conv_branches = num_conv_branches,
        )
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        y = self.depth_conv(input)
        y = self.point_conv(y)
        return y


# ----- Baseline -----
@MODELS.register(name="gcenet_baseline", arch="gcenet")
class GCENet_Baseline(nn.Module, ModelMixin):
    """Reimplement the Zero-DCE network as the baseline."""
    
    arch     : str          = "gcenet"
    name     : str          = "gcenet_baseline"
    tasks    : list[Task]   = [Task.LLE]
    mltypes  : list[MLType] = [MLType.UNSUPERVISED]
    model_dir: Path         = current_dir
    zoo      : dict         = box.Box()
    
    def __init__(self, iters: int = 8, weights: Any = None, *args, **kwargs):
        super().__init__()
        self.iters    = iters
        in_channels   = 3
        hidden_dim    = 32
        hidden_dim_x2 = hidden_dim * 2
        out_channels  = iters * 3
        self.e_conv1  = nn.Conv2d(in_channels,   hidden_dim,   3, 1, 1, bias=True)
        self.e_conv2  = nn.Conv2d(hidden_dim,    hidden_dim,   3, 1, 1, bias=True)
        self.e_conv3  = nn.Conv2d(hidden_dim,    hidden_dim,   3, 1, 1, bias=True)
        self.e_conv4  = nn.Conv2d(hidden_dim,    hidden_dim,   3, 1, 1, bias=True)
        self.e_conv5  = nn.Conv2d(hidden_dim_x2, hidden_dim,   3, 1, 1, bias=True)
        self.e_conv6  = nn.Conv2d(hidden_dim_x2, hidden_dim,   3, 1, 1, bias=True)
        self.e_conv7  = nn.Conv2d(hidden_dim_x2, out_channels, 3, 1, 1, bias=True)
        self.relu     = nn.ReLU(inplace=False)
        self.apply(weights_init)
        
        # Load weights
        self.load_weights(weights)
        
    def forward(self, image: torch.Tensor, depth: torch.Tensor = None) -> tuple[torch.Tensor, ...]:
        x  = image
        r  = self.learn_curve(x)
        rs = torch.split(r, 3, dim=1)
        y  = x
        for i in range(0, self.iters):
            y = y + rs[i] * (torch.pow(y, 2) - y)
        return r, y
    
    def learn_curve(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
        r  =    F.tanh(self.e_conv7(torch.cat([x1, x6], 1)))
        return r


# ----- Model -----
@MODELS.register(name="gcenet", arch="gcenet")
class GCENet(nn.Module, ModelMixin):
    """GCE-Net model for low-light image enhancement."""
    
    arch     : str          = "gcenet"
    name     : str          = "gcenet"
    tasks    : list[Task]   = [Task.LLE]
    mltypes  : list[MLType] = [MLType.UNSUPERVISED]
    model_dir: Path         = current_dir
    zoo      : dict         = box.Box()
    
    def __init__(
        self,
        iters    : int  = 8,
        inference: bool = False,
        weights  : Any  = None,
        *args, **kwargs
    ):
        super().__init__()
        self.iters     = iters
        self.inference = inference
        
        in_channels   = 3
        hidden_dim    = 32
        hidden_dim_x2 = hidden_dim * 2
        out_channels  = iters * 3
        self.e_conv1  = nn.Conv2d(in_channels,   hidden_dim,   3, 1, 1, bias=True)
        self.e_conv2  = nn.Conv2d(hidden_dim,    hidden_dim,   3, 1, 1, bias=True)
        self.e_conv3  = nn.Conv2d(hidden_dim,    hidden_dim,   3, 1, 1, bias=True)
        self.e_conv4  = nn.Conv2d(hidden_dim,    hidden_dim,   3, 1, 1, bias=True)
        self.e_conv5  = nn.Conv2d(hidden_dim_x2, hidden_dim,   3, 1, 1, bias=True)
        self.e_conv6  = nn.Conv2d(hidden_dim_x2, hidden_dim,   3, 1, 1, bias=True)
        self.e_conv7  = nn.Conv2d(hidden_dim_x2, out_channels, 3, 1, 1, bias=True)
        self.relu     = nn.ReLU(inplace=False)
        # self.bam      = I.BrightnessAttentionMap(gamma=2.6, kernel_size=9)
        self.gf       = I.FastGuidedFilter(kernel_size=7)
        self.bf       = kornia.filters.BilateralBlur((7, 7), 0.1, (1.5, 1.5))
        self.apply(weights_init)
        
        # Load weights
        self.load_weights(weights)
        
    def forward(self, image: torch.Tensor, depth: torch.Tensor = None) -> tuple[torch.Tensor, ...]:
        # Preprocess
        if self.inference:
            x_lr = self.interpolate_image(image, 512)
        else:
            x_lr = image
        
        # Forward
        r_lr = self.learn_curve(x_lr)
        
        # Enhancement
        rs   = torch.split(r_lr, 3, dim=1)
        y_lr = x_lr
        for i in range(0, self.iters):
            y_lr = y_lr + rs[i] * (torch.pow(y_lr, 2) - y_lr)
        # bam  = self.bam(x_lr)
        # for i in range(0, self.iters):
        #     b    = y_lr * (1 - bam)
        #     d    = y_lr * bam
        #     y_lr = b + d + rs[i] * (torch.pow(d, 2) - d)
        
        # Postprocess
        y_lr = self.bf(y_lr)
        if self.inference:
            y = self.filter_up(x_lr, y_lr, image)
        else:
            y = y_lr

        return r_lr, y
    
    def learn_curve(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], 1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], 1)))
        r  =    F.tanh(self.e_conv7(torch.cat([x1, x6], 1)))
        return r
        
    # ----- Utils -----
    def interpolate_image(self, image: torch.Tensor, size: int) -> torch.Tensor:
        """Reshapes the image based on new resolution."""
        # return F.interpolate(image, size=(down_size, down_size), mode="bicubic")
        return F.interpolate(image, size=(size, size), mode="area")
    
    def filter_up(self, x_lr: torch.Tensor, y_lr: torch.Tensor, x_hr: torch.Tensor) -> torch.Tensor:
        """Applies the guided filter to upscale the predicted image. """
        y_hr = self.gf(x_lr, y_lr, x_hr)
        y_hr = torch.clip(y_hr, 0.0, 1.0)
        return y_hr


@MODELS.register(name="gcenet_ghost", arch="gcenet")
class GCENet_Ghost(nn.Module, ModelMixin):
    """GCE-Net model for low-light image enhancement."""
    
    arch     : str          = "gcenet"
    name     : str          = "gcenet_ghost"
    tasks    : list[Task]   = [Task.LLE]
    mltypes  : list[MLType] = [MLType.UNSUPERVISED]
    model_dir: Path         = current_dir
    zoo      : dict         = box.Box()
    
    def __init__(
        self,
        iters    : int  = 8,
        inference: bool = False,
        weights  : Any  = None,
        *args, **kwargs
    ):
        super().__init__()
        self.iters     = iters
        self.inference = inference
        
        in_channels   = 3
        hidden_dim    = 32
        hidden_dim_x2 = hidden_dim * 2
        out_channels  = iters * 3
        self.e_conv1  = nn.GhostModule(in_channels,   hidden_dim,   3, relu=True)
        self.e_conv2  = nn.GhostModule(hidden_dim,    hidden_dim,   3, relu=True)
        self.e_conv3  = nn.GhostModule(hidden_dim,    hidden_dim,   3, relu=True)
        self.e_conv4  = nn.GhostModule(hidden_dim,    hidden_dim,   3, relu=True)
        self.e_conv5  = nn.GhostModule(hidden_dim_x2, hidden_dim,   3, relu=True)
        self.e_conv6  = nn.GhostModule(hidden_dim_x2, hidden_dim,   3, relu=True)
        self.e_conv7  = nn.GhostModule(hidden_dim_x2, out_channels, 3, relu=False)
        self.relu     = nn.ReLU(inplace=False)
        # self.bam      = I.BrightnessAttentionMap(gamma=2.6, kernel_size=9)
        self.gf       = I.FastGuidedFilter(kernel_size=7)
        self.bf       = kornia.filters.BilateralBlur((7, 7), 0.1, (1.5, 1.5))
        # self.apply(weights_init)
        
        # Load weights
        self.load_weights(weights)
        
    def forward(self, image: torch.Tensor, depth: torch.Tensor = None) -> tuple[torch.Tensor, ...]:
        # Preprocess
        if self.inference:
            x_lr = self.interpolate_image(image, 512)
        else:
            x_lr = image
        
        # Forward
        r_lr = self.learn_curve(x_lr)
        
        # Enhancement
        rs   = torch.split(r_lr, 3, dim=1)
        y_lr = x_lr
        for i in range(0, self.iters):
            y_lr = y_lr + rs[i] * (torch.pow(y_lr, 2) - y_lr)
        # bam  = self.bam(x_lr)
        # for i in range(0, self.iters):
        #     b    = y_lr * (1 - bam)
        #     d    = y_lr * bam
        #     y_lr = b + d + rs[i] * (torch.pow(d, 2) - d)
        
        # Postprocess
        y_lr = self.bf(y_lr)
        if self.inference:
            y = self.filter_up(x_lr, y_lr, image)
        else:
            y = y_lr

        return r_lr, y
    
    def learn_curve(self, x: torch.Tensor) -> torch.Tensor:
        x1 =        self.e_conv1(x)
        x2 =        self.e_conv2(x1)
        x3 =        self.e_conv3(x2)
        x4 =        self.e_conv4(x3)
        x5 =        self.e_conv5(torch.cat([x3, x4], 1))
        x6 =        self.e_conv6(torch.cat([x2, x5], 1))
        r  = F.tanh(self.e_conv7(torch.cat([x1, x6], 1)))
        return r
        
    # ----- Utils -----
    def interpolate_image(self, image: torch.Tensor, size: int) -> torch.Tensor:
        """Reshapes the image based on new resolution."""
        # return F.interpolate(image, size=(down_size, down_size), mode="bicubic")
        return F.interpolate(image, size=(size, size), mode="area")
    
    def filter_up(self, x_lr: torch.Tensor, y_lr: torch.Tensor, x_hr: torch.Tensor) -> torch.Tensor:
        """Applies the guided filter to upscale the predicted image. """
        y_hr = self.gf(x_lr, y_lr, x_hr)
        y_hr = torch.clip(y_hr, 0.0, 1.0)
        return y_hr
    

@MODELS.register(name="gcenet_mobileone", arch="gcenet")
class GCENet_MobileOne(nn.Module, ModelMixin):
    """GCE-Net model for low-light image enhancement."""
    
    arch     : str          = "gcenet"
    name     : str          = "gcenet_mobileone"
    tasks    : list[Task]   = [Task.LLE]
    mltypes  : list[MLType] = [MLType.UNSUPERVISED]
    model_dir: Path         = current_dir
    zoo      : dict         = {}
    
    def __init__(
        self,
        iters    : int  = 8,
        inference: bool = False,
        weights  : Any  = None,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.iters     = iters
        self.inference = inference
        
        in_channels   = 3
        hidden_dim    = 32
        hidden_dim_x2 = hidden_dim * 2
        out_channels  = iters * 3
        self.e_conv1  = MobileOneConv(in_channels,   hidden_dim,   inference,                num_conv_branches=4)
        self.e_conv2  = MobileOneConv(hidden_dim,    hidden_dim,   inference, use_se=True,   num_conv_branches=4)
        self.e_conv3  = MobileOneConv(hidden_dim,    hidden_dim,   inference, use_se=True,   num_conv_branches=4)
        self.e_conv4  = MobileOneConv(hidden_dim,    hidden_dim,   inference, use_se=True,   num_conv_branches=4)
        self.e_conv5  = MobileOneConv(hidden_dim_x2, hidden_dim,   inference, use_se=True,   num_conv_branches=4)
        self.e_conv6  = MobileOneConv(hidden_dim_x2, hidden_dim,   inference, use_se=True,   num_conv_branches=4)
        self.e_conv7  = MobileOneConv(hidden_dim_x2, out_channels, inference, use_act=False, num_conv_branches=4)
        # self.bam      = I.BrightnessAttentionMap(gamma=2.6, kernel_size=9)
        self.gf       = I.FastGuidedFilter(kernel_size=7)
        self.bf       = kornia.filters.BilateralBlur((7, 7), 0.1, (1.5, 1.5))
        
        # Load weights
        self.load_weights(weights)
        
    def forward( self, image: torch.Tensor, depth: torch.Tensor = None) -> tuple[torch.Tensor, ...]:
        # Preprocess
        if self.inference:
            x_lr = self.interpolate_image(image, 256)
        else:
            x_lr = image
        # bam = self.bam(x_lr)
        
        # Forward
        r_lr = self.learn_curve(x_lr)
        
        # Enhancement
        rs   = torch.split(r_lr, 3, dim=1)
        y_lr = x_lr
        for i in range(0, self.iters):
            y_lr = y_lr + rs[i] * (torch.pow(y_lr, 2) - y_lr)
        # bam  = self.bam(x_lr)
        # for i in range(0, self.iters):
        #     b    = y_lr * (1 - bam)
        #     d    = y_lr * bam
        #     y_lr = b + d + rs[i] * (torch.pow(d, 2) - d)
        
        # Postprocess
        y_lr = self.bf(y_lr)
        if self.inference:
            y = self.filter_up(x_lr, y_lr, image)
        else:
            y = y_lr

        return r_lr, y
    
    def learn_curve(self, x: torch.Tensor) -> torch.Tensor:
        x1 =        self.e_conv1(x)
        x2 =        self.e_conv2(x1)
        x3 =        self.e_conv3(x2)
        x4 =        self.e_conv4(x3)
        x5 =        self.e_conv5(torch.cat([x3, x4], 1))
        x6 =        self.e_conv6(torch.cat([x2, x5], 1))
        r  = F.tanh(self.e_conv7(torch.cat([x1, x6], 1)))
        return r
        
    # ----- Utils -----
    def interpolate_image(self, image: torch.Tensor, size: int) -> torch.Tensor:
        """Reshapes the image based on new resolution."""
        # return F.interpolate(image, size=(down_size, down_size), mode="bicubic")
        return F.interpolate(image, size=(size, size), mode="area")
    
    def filter_up(self, x_lr: torch.Tensor, y_lr: torch.Tensor, x_hr: torch.Tensor) -> torch.Tensor:
        """Applies the guided filter to upscale the predicted image. """
        y_hr = self.gf(x_lr, y_lr, x_hr)
        y_hr = torch.clip(y_hr, 0.0, 1.0)
        return y_hr
