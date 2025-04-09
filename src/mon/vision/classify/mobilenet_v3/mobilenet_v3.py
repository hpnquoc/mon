#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements MobileNetV3 models."""

__all__ = [
    "MobileNetV3Large",
    "MobileNetV3Small",
]

from abc import ABC

from torchvision.models import mobilenet_v3_large, mobilenet_v3_small

from mon import core, nn
from mon.constants import MLType, MODELS, ZOO_DIR
from mon.vision.classify import base

current_file = core.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# ----- Model -----
class MobileNetV3(nn.ExtraModel, base.ImageClassificationModel, ABC):
    """MobileNetV3 model for image classification.

    References:
        - https://arxiv.org/abs/1905.02244
    """
    
    arch     : str          = "mobilenet"
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
    

@MODELS.register(name="mobilenet_v3_large", arch="mobilenet")
class MobileNetV3Large(MobileNetV3):
    """MobileNetV3-Large model for image classification.

    Args:
        num_classes: Number of output classes. Default is ``1000``.
        dropout: Dropout rate for the model. Default is ``0.2``.
    """
    
    name: str  = "mobilenet_v3_large"
    zoo : dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth",
            "path"       : ZOO_DIR / "vision/classify/mobilenet/mobilenet_v3_large/imagenet1k_v1/mobilenet_v3_large_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
        "imagenet1k_v2": {
            "url"        : "https://download.pytorch.org/models/mobilenet_v3_large-5c1a4163.pth",
            "path"       : ZOO_DIR / "vision/classify/mobilenet/mobilenet_v3_large/imagenet1k_v2/mobilenet_v3_large_imagenet1k_v2.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(self, num_classes: int = 1000, dropout: float = 0.2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_classes = self.parse_num_classes(num_classes)
        
        # Network
        self.model = mobilenet_v3_large(num_classes=num_classes, dropout=dropout)
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)


@MODELS.register(name="mobilenet_v3_small", arch="mobilenet")
class MobileNetV3Small(MobileNetV3):
    """MobileNetV3-Small model for image classification.

    Args:
        num_classes: Number of output classes. Default is ``1000``.
        dropout: Dropout rate for the model. Default is ``0.2``.
    """
    
    name: str  = "mobilenet_v3_small"
    zoo : dict = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth",
            "path"       : ZOO_DIR / "vision/classify/mobilenet/mobilenet_v3_small/imagenet1k_v1/mobilenet_v3_small_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(self, num_classes: int = 1000, dropout: float = 0.2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_classes = self.parse_num_classes(num_classes)
        
        # Network
        self.model = mobilenet_v3_small(num_classes=num_classes, dropout=dropout)
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)
