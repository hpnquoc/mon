#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements VGG models."""

__all__ = [
    "VGG11",
    "VGG11_BN",
    "VGG13",
    "VGG13_BN",
    "VGG16",
    "VGG16_BN",
    "VGG19",
    "VGG19_BN",
]

from abc import ABC

from torchvision.models import (
    vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19, vgg19_bn,
)

from mon import core, nn
from mon.constants import MLType, MODELS, ZOO_DIR
from mon.vision.classify import base

current_file = core.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# ----- VGG -----
class VGG(nn.ExtraModel, base.ImageClassificationModel, ABC):
    """VGG model for image classification.

    References:
        - https://arxiv.org/abs/1409.1556
    """
    
    arch     : str          = "vgg"
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


@MODELS.register(name="vgg11", arch="vgg")
class VGG11(VGG):
    """VGG-11 model for image classification.

    Args:
        num_classes: Number of output classes. Default is ``1000``.
        dropout: Dropout rate for the model. Default is ``0.5``.
    """
    
    name: str  = "vgg11"
    zoo : dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/vgg11-8a719046.pth",
            "path"       : ZOO_DIR / "vision/classify/vgg/vgg11/imagenet1k_v1/vgg11_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_classes = self.parse_num_classes(num_classes)
        
        # Network
        self.model = vgg11(num_classes=num_classes, dropout=dropout)
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)


@MODELS.register(name="vgg13", arch="vgg")
class VGG13(VGG):
    """VGG-13 model for image classification.

    Args:
        num_classes: Number of output classes. Default is ``1000``.
        dropout: Dropout rate for the model. Default is ``0.5``.
    """
    
    name: str  = "vgg13"
    zoo : dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/vgg13-19584684.pth",
            "path"       : ZOO_DIR / "vision/classify/vgg/vgg13/imagenet1k_v1/vgg13_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_classes = self.parse_num_classes(num_classes)
        
        # Network
        self.model = vgg13(num_classes=num_classes, dropout=dropout)
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)
           

@MODELS.register(name="vgg16", arch="vgg")
class VGG16(VGG):
    """VGG-16 model for image classification.

    Args:
        num_classes: Number of output classes. Default is ``1000``.
        dropout: Dropout rate for the model. Default is ``0.5``.
    """
    
    name: str  = "vgg16"
    zoo : dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/vgg16-397923af.pth",
            "path"       : ZOO_DIR / "vision/classify/vgg/vgg16/imagenet1k_v1/vgg16_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_classes = self.parse_num_classes(num_classes)
        
        # Network
        self.model = vgg16(num_classes=num_classes, dropout=dropout)
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)


@MODELS.register(name="vgg19", arch="vgg")
class VGG19(VGG):
    """VGG-19 model for image classification.

    Args:
        num_classes: Number of output classes. Default is ``1000``.
        dropout: Dropout rate for the model. Default is ``0.5``.
    """
    
    name: str  = "vgg19"
    zoo : dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth",
            "path"       : ZOO_DIR / "vision/classify/vgg/vgg19/imagenet1k_v1/vgg19_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_classes = self.parse_num_classes(num_classes)
        
        # Network
        self.model = vgg19(num_classes=num_classes, dropout=dropout)
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)


# ----- VGG-BN-----
@MODELS.register(name="vgg11_bn", arch="vgg")
class VGG11_BN(VGG):
    """VGG-11-BN model for image classification.

    Args:
        num_classes: Number of output classes. Default is ``1000``.
        dropout: Dropout rate for the model. Default is ``0.5``.
    """
    
    name: str  = "vgg11_bn"
    zoo : dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/vgg11_bn-6002323d.pth",
            "path"       : ZOO_DIR / "vision/classify/vgg/vgg11_bn/imagenet1k_v1/vgg11_bn_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_classes = self.parse_num_classes(num_classes)
        
        # Network
        self.model = vgg11_bn(num_classes=num_classes, dropout=dropout)
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)
            
            
@MODELS.register(name="vgg13_bn", arch="vgg")
class VGG13_BN(VGG):
    """VGG-13-BN model for image classification.

    Args:
        num_classes: Number of output classes. Default is ``1000``.
        dropout: Dropout rate for the model. Default is ``0.5``.
    """
    
    name: str  = "vgg13_bn"
    zoo : dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/vgg13_bn-abd245e5.pth",
            "path"       : ZOO_DIR / "vision/classify/vgg/vgg13_bn/imagenet1k_v1/vgg13_bn_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_classes = self.parse_num_classes(num_classes)
        
        # Network
        self.model = vgg13_bn(num_classes=num_classes, dropout=dropout)
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)
            
            
@MODELS.register(name="vgg16_bn", arch="vgg")
class VGG16_BN(VGG):
    """VGG-16-BN model for image classification.

    Args:
        num_classes: Number of output classes. Default is ``1000``.
        dropout: Dropout rate for the model. Default is ``0.5``.
    """
    
    name: str  = "vgg16_bn"
    zoo : dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/vgg16_bn-6c64b313.pth",
            "path"       : ZOO_DIR / "vision/classify/vgg/vgg16_bn/imagenet1k_v1/vgg16_bn_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_classes = self.parse_num_classes(num_classes)
        
        # Network
        self.model = vgg16_bn(num_classes=num_classes, dropout=dropout)
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)


@MODELS.register(name="vgg19_bn", arch="vgg")
class VGG19_BN(VGG):
    """VGG-19-BN model for image classification.

    Args:
        num_classes: Number of output classes. Default is ``1000``.
        dropout: Dropout rate for the model. Default is ``0.5``.
    """
    
    name: str  = "vgg19_bn"
    zoo : dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/vgg19_bn-c79401a0.pth",
            "path"       : ZOO_DIR / "vision/classify/vgg/vgg19_bn/imagenet1k_v1/vgg19_bn_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_classes = self.parse_num_classes(num_classes)
        
        # Network
        self.model = vgg19_bn(num_classes=num_classes, dropout=dropout)
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)
