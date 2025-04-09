#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements Swin Transformer models."""

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
from mon.constants import MLType, MODELS, ZOO_DIR
from mon.vision.classify import base

current_file = core.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# ----- Model -----
class SwinTransformer(nn.ExtraModel, base.ImageClassificationModel, ABC):
    """Swin Transformer model for image classification.

    References:
        - https://arxiv.org/pdf/2103.14030
    """
    
    arch     : str          = "swin"
    mltypes  : list[MLType] = [MLType.SUPERVISED]
    model_dir: core.Path    = current_dir
    zoo      : dict         = {}
    
    # ----- Initialize -----
    def init_weights(self, m: nn.Module):
        """Initializes weights for the model.
    
        Args:
            m: ``nn.Module`` to initialize weights for.
        """
        pass

    # ----- Forward Pass -----
    def forward(self, datapoint: dict, *args, **kwargs) -> dict:
        """Performs forward pass on the model.
    
        Args:
            datapoint: ``dict`` with image data.
    
        Returns:
            ``dict`` of predictions with ``"logits"`` keys.
        """
        x = datapoint["image"]
        y = self.model(x)
        return {"logits": y}
    

@MODELS.register(name="swin_t", arch="swin")
class Swin_T(SwinTransformer):
    """Swin-T model for image classification.

    Args:
        num_classes: Number of output classes. Default is ``1000``.
        dropout: Dropout rate for the model. Default is ``0.0``.
        attention_dropout: Attention dropout rate. Default is ``0.0``.
    """
    
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
    """Swin-S model for image classification.

    Args:
        num_classes: Number of output classes. Default is ``1000``.
        dropout: Dropout rate for the model. Default is ``0.0``.
        attention_dropout: Attention dropout rate. Default is ``0.0``.
    """
    
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
    """Swin-B model for image classification.

    Args:
        num_classes: Number of output classes. Default is ``1000``.
        dropout: Dropout rate for the model. Default is ``0.0``.
        attention_dropout: Attention dropout rate. Default is ``0.0``.
    """
    
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
    """Swin-V2-T model for image classification.

    Args:
        num_classes: Number of output classes. Default is ``1000``.
        dropout: Dropout rate for the model. Default is ``0.0``.
        attention_dropout: Attention dropout rate. Default is ``0.0``.
    """
    
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
    """Swin-V2-S model for image classification.

    Args:
        num_classes: Number of output classes. Default is ``1000``.
        dropout: Dropout rate for the model. Default is ``0.0``.
        attention_dropout: Attention dropout rate. Default is ``0.0``.
    """
    
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
    """Swin-V2-B model for image classification.

    Args:
        num_classes: Number of output classes. Default is ``1000``.
        dropout: Dropout rate for the model. Default is ``0.0``.
        attention_dropout: Attention dropout rate. Default is ``0.0``.
    """
    
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
