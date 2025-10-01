#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements miscellaneous layers."""

__all__ = [
    "ChannelShuffle",
    "Embedding",
    "EmbeddingBag",
    "Flatten",
    "Fold",
    "PixelShuffle",
    "PixelUnshuffle",
    "Unflatten",
    "Unfold",
    "Upsample",
    "UpsamplingBilinear2d",
    "UpsamplingNearest2d",
]

from .flatten import Flatten, Unflatten
from .fold import Fold, Unfold
from .shuffle import ChannelShuffle, PixelShuffle, PixelUnshuffle
from .sparse import Embedding, EmbeddingBag
from .upsampling import Upsample, UpsamplingBilinear2d, UpsamplingNearest2d
