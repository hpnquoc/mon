#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements Inception v3 models."""

__all__ = [
    "Inception3",
]

from torchvision.models import inception_v3

from mon import core, nn
from mon.constants import MLType, MODELS, ZOO_DIR
from mon.vision.classify import base

current_file = core.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# ----- Model -----
@MODELS.register(name="inception_v3", arch="inception")
class Inception3(nn.ExtraModel, base.ImageClassificationModel):
    """Inception v3 model for image classification.

    Args:
        num_classes: Number of output classes. Default is ``1000``.
        aux_logits: Use auxiliary logits if True. Default is ``True``.
        dropout: Dropout rate for the model. Default is ``0.5``
    Notes:
        Expects input tensors of size ``N x 3 x 299 x 299``.
   
    References:
        - https://arxiv.org/abs/1512.00567
    """
    
    arch     : str          = "inception"
    name     : str          = "inception_v3"
    mltypes  : list[MLType] = [MLType.SUPERVISED]
    model_dir: core.Path    = current_dir
    zoo      : dict         = {
        "imagenet1k_v1": {
            "url"        : "https://download.pytorch.org/models/inception_v3_google-0cc3c7bd.pth",
            "path"       : ZOO_DIR / "vision/classify/inception/inception_v3/imagenet1k_v1/inception_v3_imagenet1k_v1.pth",
            "num_classes": 1000,
        },
    }
    
    def __init__(
        self,
        num_classes: int   = 1000,
        aux_logits : bool  = True,
        dropout    : float = 0.5,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        num_classes = self.parse_num_classes(num_classes)
        
        # Network
        self.model = inception_v3(
            num_classes = num_classes,
            aux_logits  = aux_logits,
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
