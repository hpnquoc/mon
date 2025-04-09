#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements ConvNeXt models."""

__all__ = [
    "ConvNeXtBase",
    "ConvNeXtLarge",
    "ConvNeXtSmall",
    "ConvNeXtTiny",
]

from abc import ABC

from torchvision.models import (
    convnext_base, convnext_large, convnext_small, convnext_tiny,
)

from mon import core, nn
from mon.constants import MLType, MODELS, ZOO_DIR
from mon.vision.classify import base

current_file = core.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# ----- Model -----
class ConvNeXt(nn.ExtraModel, base.ImageClassificationModel, ABC):
    """ConvNeXt model for image classification.

    References:
        - https://arxiv.org/abs/2201.03545
    """
    
    arch     : str          = "convnext"
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


@MODELS.register(name="convnext_base", arch="convnext")
class ConvNeXtBase(ConvNeXt):
    """ConvNeXt Base model for image classification.
    
    Args:
        num_classes: Number of output classes. Default is ``1000``.
    """
    
    name: str = "convnext_base"
    zoo : dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/convnext_base-6075fbad.pth",
            "path"       : ZOO_DIR / "vision/classify/convnext/convnext_base/imagenet1k_v1/convnext_base_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(self, num_classes: int = 1000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_classes = self.parse_num_classes(num_classes)
        
        # Network
        self.model = convnext_base(num_classes=num_classes)
        
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)


@MODELS.register(name="convnext_tiny", arch="convnext")
class ConvNeXtTiny(ConvNeXt):
    """ConvNeXt Tiny model for image classification.

    Args:
        num_classes: Number of output classes. Default is 1000.
    """
    
    name: str = "convnext_tiny"
    zoo : dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/convnext_tiny-983f1562.pth",
            "path"       : ZOO_DIR / "vision/classify/convnext/convnext_tiny/imagenet1k_v1/convnext_tiny_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(self, num_classes: int = 1000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_classes = self.parse_num_classes(num_classes)
        
        # Network
        self.model = convnext_tiny(num_classes=num_classes)
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)


@MODELS.register(name="convnext_small", arch="convnext")
class ConvNeXtSmall(ConvNeXt):
    """ConvNeXt Small model for image classification.

    Args:
        num_classes: Number of output classes. Default is ``1000``.
    """
    
    name: str  = "convnext_small"
    zoo : dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/convnext_small-0c510722.pth",
            "path"       : ZOO_DIR / "vision/classify/convnext/convnext_small/imagenet1k_v1/convnext_small_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(self, num_classes: int = 1000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_classes = self.parse_num_classes(num_classes)
        
        # Network
        self.model = convnext_small(num_classes=num_classes)
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)


@MODELS.register(name="convnext_large", arch="convnext")
class ConvNeXtLarge(ConvNeXt):
    """ConvNeXt Large model for image classification.

    Args:
        num_classes: Number of output classes. Default is ``1000``.
    """
    
    name: str  = "convnext_large"
    zoo : dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/convnext_large-ea097f82.pth",
            "path"       : ZOO_DIR / "vision/classify/convnext/convnext_large/imagenet1k_v1/convnext_large_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(self, num_classes: int = 1000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_classes = self.parse_num_classes(num_classes)
        
        # Network
        self.model = convnext_large(num_classes=num_classes)
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)
