#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Registers ``torch``'s loss functions."""

__all__ = [
    "BCELoss",
    "BCEWithLogitsLoss",
    "CTCLoss",
    "CosineEmbeddingLoss",
    "CrossEntropyLoss",
    "GaussianNLLLoss",
    "HingeEmbeddingLoss",
    "HuberLoss",
    "KLDivLoss",
    "L1Loss",
    "MSELoss",
    "MarginRankingLoss",
    "MultiLabelMarginLoss",
    "MultiLabelSoftMarginLoss",
    "MultiMarginLoss",
    "NLLLoss",
    "NLLLoss2d",
    "PoissonNLLLoss",
    "SmoothL1Loss",
    "SoftMarginLoss",
    "TripletMarginLoss",
    "TripletMarginWithDistanceLoss",
]

from torch.nn.modules.loss import *  # Expose all losses from ``torch.nn.modules.loss``
