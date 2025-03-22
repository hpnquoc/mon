#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Swin Transformer.

This module implements Swin Transformer models.
"""

from __future__ import annotations

__all__ = [
    "Swin_B",
    "Swin_S",
    "Swin_T",
    "Swin_V2_B",
    "Swin_V2_S",
    "Swin_V2_T",
]

from abc import ABC

from torchvision.models import (
    swin_b, swin_s, swin_t, swin_v2_b, swin_v2_s, swin_v2_t,
)

from mon import core, nn
from mon.globals import MODELS, Scheme, ZOO_DIR
from mon.vision.classify import base

console      = core.console
current_file = core.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# region Model

class SwinTransformer(nn.ExtraModel, base.ImageClassificationModel, ABC):
    """Implements Swin Transformer from the paper: "Swin Transformer:
    Hierarchical Vision Transformer using Shifted Windows".
    
    References:
        https://arxiv.org/pdf/2103.14030
    """
    
    arch     : str          = "swin"
    schemes  : list[Scheme] = [Scheme.SUPERVISED]
    model_dir: core.Path    = current_dir
    zoo      : dict         = {}
    
    def init_weights(self, m: nn.Module):
        pass
    
    def forward(self, datapoint: dict, *args, **kwargs) -> dict:
        x = datapoint["image"]
        y = self.model(x)
        return {"logits": y}
    

@MODELS.register(name="swin_t", arch="swin")
class Swin_T(SwinTransformer):
    
    name: str  = "swin_t"
    zoo : dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/swin_t-704ceda3.pth",
            "path"       : ZOO_DIR / "vision/classify/swin/swin_t/imagenet1k_v1/swin_t_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(
        self,
        num_classes      : int   = 1000,
        dropout          : float = 0.0,
        attention_dropout: float = 0.0,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        num_classes = self.parse_num_classes(num_classes)
        
        # Network
        self.model = swin_t(
            num_classes       = num_classes,
            dropout           = dropout,
            attention_dropout = attention_dropout,
        )
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)


@MODELS.register(name="swin_s", arch="swin")
class Swin_S(SwinTransformer):
    
    name: str  = "swin_s"
    zoo : dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/swin_s-5e29d889.pth",
            "path"       : ZOO_DIR / "vision/classify/swin/swin_s/imagenet1k_v1/swin_s_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(
        self,
        num_classes      : int   = 1000,
        dropout          : float = 0.0,
        attention_dropout: float = 0.0,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        num_classes = self.parse_num_classes(num_classes)
        
        # Network
        self.model = swin_s(
            num_classes       = num_classes,
            dropout           = dropout,
            attention_dropout = attention_dropout,
        )
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)


@MODELS.register(name="swin_b", arch="swin")
class Swin_B(SwinTransformer):
    
    name: str  = "swin_b"
    zoo : dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/swin_b-68c6b09e.pth",
            "path"       : ZOO_DIR / "vision/classify/swin/swin_b/imagenet1k_v1/swin_b_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(
        self,
        num_classes      : int   = 1000,
        dropout          : float = 0.0,
        attention_dropout: float = 0.0,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        num_classes = self.parse_num_classes(num_classes)
        
        # Network
        self.model = swin_b(
            num_classes       = num_classes,
            dropout           = dropout,
            attention_dropout = attention_dropout,
        )
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)
        

@MODELS.register(name="swin_v2_t", arch="swin")
class Swin_V2_T(SwinTransformer):
    
    name: str  = "swin_v2_t"
    zoo : dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/swin_v2_t-b137f0e2.pth",
            "path"       : ZOO_DIR / "vision/classify/swin/swin_v2_t/imagenet1k_v1/swin_v2_t_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(
        self,
        num_classes      : int   = 1000,
        dropout          : float = 0.0,
        attention_dropout: float = 0.0,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        num_classes = self.parse_num_classes(num_classes)
        
        # Network
        self.model = swin_v2_t(
            num_classes       = num_classes,
            dropout           = dropout,
            attention_dropout = attention_dropout,
        )
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)
        

@MODELS.register(name="swin_v2_s", arch="swin")
class Swin_V2_S(SwinTransformer):
    
    name: str  = "swin_v2_s"
    zoo : dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/swin_v2_s-637d8ceb.pth",
            "path"       : ZOO_DIR / "vision/classify/swin/swin_v2_s/imagenet1k_v1/swin_v2_s_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(
        self,
        num_classes      : int   = 1000,
        dropout          : float = 0.0,
        attention_dropout: float = 0.0,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        num_classes = self.parse_num_classes(num_classes)
        
        # Network
        self.model = swin_v2_s(
            num_classes       = num_classes,
            dropout           = dropout,
            attention_dropout = attention_dropout,
        )
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)
        

@MODELS.register(name="swin_v2_b", arch="swin")
class Swin_V2_B(SwinTransformer):
    
    name: str  = "swin_v2_b"
    zoo : dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/swin_v2_b-781e5279.pth",
            "path"       : ZOO_DIR / "vision/classify/swin/swin_v2_b/imagenet1k_v1/swin_v2_b_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(
        self,
        num_classes      : int   = 1000,
        dropout          : float = 0.0,
        attention_dropout: float = 0.0,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        num_classes = self.parse_num_classes(num_classes)
        
        # Network
        self.model = swin_v2_b(
            num_classes       = num_classes,
            dropout           = dropout,
            attention_dropout = attention_dropout,
        )
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)
        
# endregion
