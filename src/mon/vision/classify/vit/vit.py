#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements ViT (Vision Transformer) models."""

__all__ = [
    "ViT_B_16",
    "ViT_B_32",
    "ViT_H_14",
    "ViT_L_16",
    "ViT_L_32",
]

from abc import ABC

from torchvision.models import vit_b_16, vit_b_32, vit_h_14, vit_l_16, vit_l_32

from mon import core, nn
from mon.constants import MLType, MODELS, ZOO_DIR
from mon.vision.classify import base

current_file = core.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# ----- Model -----
class VisionTransformer(base.ImageClassificationModel, ABC):
    """Vision Transformer model for image classification.

    References:
        - https://arxiv.org/abs/2010.11929
    """
    
    arch     : str          = "vit"
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


@MODELS.register(name="vit_b_16", arch="vit")
class ViT_B_16(VisionTransformer):
    """ViT-B/16 model for image classification.

    Args:
        num_classes: Number of output classes. Default is ``1000``.
        dropout: Dropout rate for the model. Default is ``0.0``.
        attention_dropout: Attention dropout rate. Default is ``0.0``.
    """
    
    name: str  = "vit_b_16"
    zoo : dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/vit_b_16-c867db91.pth",
            "path"       : ZOO_DIR / "vision/classify/vit/vit_b_16/imagenet1k_v1/vit_b_16_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
        "imagenet1k_swag_e2e_v1": {
            "url"        : "https://download.pytorch.org/models/vit_b_16_swag-9ac1b537.pth",
            "path"       : ZOO_DIR / "vision/classify/vit/vit_b_16_swag/imagenet1k_v1/vit_b_16_swag_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
        "imagenet1k_swag_linear_v1": {
            "url"        : "https://download.pytorch.org/models/vit_b_16_lc_swag-4e70ced5.pth",
            "path"       : ZOO_DIR / "vision/classify/vit/vit_b_16_lc_swag/imagenet1k_v1/vit_b_16_lc_swag_imagenet1k_v1.pth",
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
        self.model = vit_b_16(
            num_classes       = num_classes,
            dropout           = dropout,
            attention_dropout = attention_dropout,
        )
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)


@MODELS.register(name="vit_b_32", arch="vit")
class ViT_B_32(VisionTransformer):
    """ViT-B/32 model for image classification.

    Args:
        num_classes: Number of output classes. Default is ``1000``.
        dropout: Dropout rate for the model. Default is ``0.0``.
        attention_dropout: Attention dropout rate. Default is ``0.0``.
    """
    
    name: str  = "vit_b_32"
    zoo : dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/vit_b_32-d86f8d99.pth",
            "path"       : ZOO_DIR / "vision/classify/vit/vit_b_32/imagenet1k_v1/vit_b_32_imagenet1k_v1.pth",
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
        self.model = vit_b_32(
            num_classes       = num_classes,
            dropout           = dropout,
            attention_dropout = attention_dropout,
        )
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)


@MODELS.register(name="vit_l_16", arch="vit")
class ViT_L_16(VisionTransformer):
    """ViT-L/16 model for image classification.

    Args:
        num_classes: Number of output classes. Default is ``1000``.
        dropout: Dropout rate for the model. Default is ``0.0``.
        attention_dropout: Attention dropout rate. Default is ``0.0``.
    """
    
    name: str  = "vit_l_16"
    zoo : dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/vit_l_16-852ce7e3.pth",
            "path"       : ZOO_DIR / "vision/classify/vit/vit_l_16/imagenet1k_v1/vit_l_16_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
        "imagenet1k_swag_e2e_v1": {
            "url"        : "https://download.pytorch.org/models/vit_l_16_swag-4f3808c9.pth",
            "path"       : ZOO_DIR / "vision/classify/vit/vit_l_16_swag/imagenet1k_v1/vit_l_16_swag_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
        "imagenet1k_swag_linear_v1": {
            "url"        : "https://download.pytorch.org/models/vit_l_16_lc_swag-4d563306.pth",
            "path"       : ZOO_DIR / "vision/classify/vit/vit_l_16_lc_swag/imagenet1k_v1/vit_l_16_lc_swag_imagenet1k_v1.pth",
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
        self.model = vit_l_16(
            num_classes       = num_classes,
            dropout           = dropout,
            attention_dropout = attention_dropout,
        )
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)


@MODELS.register(name="vit_l_32", arch="vit")
class ViT_L_32(VisionTransformer):
    """ViT-L/32 model for image classification.

    Args:
        num_classes: Number of output classes. Default is ``1000``.
        dropout: Dropout rate for the model. Default is ``0.0``.
        attention_dropout: Attention dropout rate. Default is ``0.0``.
    """
    
    name: str  = "vit_l_32"
    zoo : dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/vit_l_32-c7638314.pth",
            "path"       : ZOO_DIR / "vision/classify/vit/vit_l_32/imagenet1k_v1/vit_l_32_imagenet1k_v1.pth",
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
        self.model = vit_l_32(
            num_classes       = num_classes,
            dropout           = dropout,
            attention_dropout = attention_dropout,
        )
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)
      

@MODELS.register(name="vit_h_14", arch="vit")
class ViT_H_14(VisionTransformer):
    """ViT-H/14 model for image classification.

    Args:
        num_classes: Number of output classes. Default is ``1000``.
        dropout: Dropout rate for the model. Default is ``0.0``.
        attention_dropout: Attention dropout rate. Default is ``0.0``.
    """
    
    name: str  = "vit_h_14"
    zoo : dict = {
        "imagenet1k_swag_e2e_v1": {
            "url"        : "https://download.pytorch.org/models/vit_h_14_swag-80465313.pth",
            "path"       : ZOO_DIR / "vision/classify/vit/vit_h_14_swag/imagenet1k_v1/vit_h_14_swag_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
        "imagenet1k_swag_linear_v1": {
            "url"        : "https://download.pytorch.org/models/vit_h_14_lc_swag-c1eb923e.pth",
            "path"       : ZOO_DIR / "vision/classify/vit/vit_h_14_lc_swag/imagenet1k_v1/vit_h_14_lc_swag_imagenet1k_v1.pth",
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
        self.model = vit_h_14(
            num_classes       = num_classes,
            dropout           = dropout,
            attention_dropout = attention_dropout,
        )
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)
