#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""DepthAnything.

This module implements a wrapper for DepthAnything models. It provides a
simple interface for loading pre-trained models and performing inference.

Notice:
This is the first example of using a third-party package in the `mon` package.
Why? Because reimplementing all of :obj:`depth_anything_v2` is a lot of work and
is not a smart idea.
"""

from __future__ import annotations

__all__ = [
    "DepthAnything_ViTB",
    "DepthAnything_ViTL",
    "DepthAnything_ViTS",
    "build_depth_anything",
]

from abc import ABC
from typing import Any, Literal

from depth_anything import dpt
from mon import core, nn
from mon.globals import MODELS, Scheme, ZOO_DIR
from mon.vision.depth import base

console       = core.console
error_console = core.error_console
current_file  = core.Path(__file__).absolute()
current_dir   = current_file.parents[0]


# region Model

class DepthAnything(nn.ExtraModel, base.DepthEstimationModel, ABC):
    """This class implements a wrapper for :obj:`DepthAnythingV2` models
    defined in :obj:`mon_extra.vision.depth.depth_anything_v2`.
    """
    
    model_dir: core.Path    = current_dir
    arch     : str          = "depth_anything"
    schemes  : list[Scheme] = [Scheme.INFERENCE]
    zoo      : dict         = {}
    
    def init_weights(self, m: nn.Module):
        pass
    
    def forward(self, datapoint: dict, *args, **kwargs) -> dict:
        self.assert_datapoint(datapoint)
        x = datapoint.get("image")
        y = self.model(x)
        y = (y - y.min()) / (y.max() - y.min())  # Normalize the depth map in the range [0, 1].
        y = y.unsqueeze(1)
        return {"depth": y}


@MODELS.register(name="depth_anything_vits", arch="depth_anything")
class DepthAnything_ViTS(DepthAnything):
    
    zoo: dict = {
        "pretrained": {
            "url"        : None,
            "path"       : ZOO_DIR / "vision/depth/depth_anything/depth_anything_vits/pretrained/depth_anything_vits.pth",
            "num_classes": None,
        },
    }
    
    def __init__(
        self,
        name       : str = "depth_anything_vits",
        in_channels: int = 3,
        weights    : Any = "pretrained",
        *args, **kwargs
    ):
        super().__init__(
            name        = name,
            in_channels = in_channels,
            weights     = weights,
            *args, **kwargs
        )
        self.in_channels = in_channels or self.in_channels
        self.model       = dpt.DepthAnything(
            config = {
                "encoder"     : "vits",
                "features"    : 64,
                "out_channels": [48, 96, 192, 384],
            },
        )
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)


@MODELS.register(name="depth_anything_vitb", arch="depth_anything")
class DepthAnything_ViTB(DepthAnything):
    
    zoo: dict = {
        "pretrained": {
            "url"        : None,
            "path"       : ZOO_DIR / "vision/depth/depth_anything/depth_anything_vitb/pretrained/depth_anything_vitb.pth",
            "num_classes": None,
        },
    }

    def __init__(
        self,
        name       : str = "depth_anything_vits",
        in_channels: int = 3,
        weights    : Any = "pretrained",
        *args, **kwargs
    ):
        super().__init__(
            name        = name,
            in_channels = in_channels,
            weights     = weights,
            *args, **kwargs
        )
        self.in_channels = in_channels or self.in_channels
        self.model       = dpt.DepthAnything(
            config = {
                "encoder"     : "vitb",
                "features"    : 128,
                "out_channels": [96, 192, 384, 768],
            },
        )
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)
            

@MODELS.register(name="depth_anything_vitl", arch="depth_anything")
class DepthAnything_ViTL(DepthAnything):
    
    zoo: dict = {
        "da_2k": {
            "url"        : None,
            "path"       : ZOO_DIR / "vision/depth/depth_anything/depth_anything_vitl/pretrained/depth_anything_vitl.pth",
            "num_classes": None,
        },
    }

    def __init__(
        self,
        name       : str = "depth_anything_vits",
        in_channels: int = 3,
        weights    : Any = "pretrained",
        *args, **kwargs
    ):
        super().__init__(
            name        = name,
            in_channels = in_channels,
            weights     = weights,
            *args, **kwargs
        )
        self.in_channels = in_channels or self.in_channels
        self.model       = dpt.DepthAnything(
            config = {
                "encoder"     : "vitl",
                "features"    : 256,
                "out_channels": [256, 512, 1024, 1024],
            },
        )
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)


def build_depth_anything(
    encoder     : Literal["vits", "vitb", "vitl"] = "vits",
    in_channels : int = 3,
    weights     : Any = "pretrained",
    *args, **kwargs
) -> DepthAnything:
    if encoder not in ["vits", "vitb", "vitl"]:
        raise ValueError(f"`encoder` must be one of ['vits', 'vitb', 'vitl'], but got {encoder}.")
    if encoder == "vits":
        return DepthAnything_ViTS(in_channels=in_channels, weights=weights, *args, **kwargs)
    elif encoder == "vitb":
        return DepthAnything_ViTB(in_channels=in_channels, weights=weights, *args, **kwargs)
    elif encoder == "vitl":
        return DepthAnything_ViTL(in_channels=in_channels, weights=weights, *args, **kwargs)
        
# endregion
