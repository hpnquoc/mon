#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements MobileNetV2 models."""

__all__ = [
    "MobileNetV2",
]

from torchvision.models import mobilenet_v2

from mon import core, nn
from mon.constants import MLType, MODELS, ZOO_DIR
from mon.vision.classify import base

current_file = core.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# ----- Model -----
@MODELS.register(name="mobilenet_v2", arch="mobilenet")
class MobileNetV2(nn.ExtraModel, base.ImageClassificationModel):
    """MobileNetV2 model for image classification.

    Args:
        num_classes: Number of output classes. Default is ``1000``.
        width_mult: Width multiplier for the network. Default is ``1.0``.
        dropout: Dropout rate for the model. Default is ``0.2``.
    
    References:
        - https://arxiv.org/abs/1801.04381
    """
    
    arch     : str          = "mobilenet"
    name     : str          = "mobilenet_v2"
    mltypes  : list[MLType] = [MLType.SUPERVISED]
    model_dir: core.Path    = current_dir
    zoo      : dict         = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/mobilenet_v2-b0353104.pth",
            "path"       : ZOO_DIR / "vision/classify/mobilenet/mobilenet_v2/imagenet1k_v1/mobilenet_v2_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
        "imagenet1k_v2": {
            "url"        : "https://download.pytorch.org/models/mobilenet_v2-7ebf99e0.pth",
            "path"       : ZOO_DIR / "vision/classify/mobilenet/mobilenet_v2/imagenet1k_v2/mobilenet_v2_imagenet1k_v2.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(
        self,
        num_classes: int   = 1000,
        width_mult : float = 1.0,
        dropout    : float = 0.2,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        num_classes = self.parse_num_classes(num_classes)
        
        # Network
        self.model = mobilenet_v2(
            num_classes = num_classes,
            width_mult  = width_mult,
            dropout     = dropout,
        )
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)
    
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
