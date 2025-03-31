#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements linearity layers."""

from __future__ import annotations

__all__ = [
    "Bilinear",
    "Identity",
    "LazyLinear",
    "Linear",
]

from torch.nn.modules.linear import Bilinear, Identity, LazyLinear, Linear
