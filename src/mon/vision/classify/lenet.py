#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements LeNet models."""

from __future__ import annotations

__all__ = [
    "LeNet",
]

from mon.globals import MODELS
from mon.vision import core
from mon.vision.classify import base

console      = core.console
_current_dir = core.Path(__file__).absolute().parent


# region Model

@MODELS.register(name="lenet")
class LeNet(base.ImageClassificationModel):
    """LeNet.
    
    See Also: :class:`mon.vision.enhance.base.ImageEnhancementModel`
    """
    
    configs     = {}
    zoo         = {}
    map_weights = {}
    
    def __init__(self, *args, **kwargs):
        kwargs |= {
            "config" : "lenet.yaml",
            "name"   : "lenet",
            "variant": "lenet"
        }
        super().__init__(*args, **kwargs)
    
# endregion
