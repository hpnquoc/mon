#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements the paper: "Zero-Shot Noise2Noise: Efficient Image Denoising without any
Data," CVPR 2023.

References:
    - Code: https://colab.research.google.com/drive/1i82nyizTdszyHkaHBuKPbWnTzao8HF9b?usp=sharing#scrollTo=Srf0GQTYrkxA
"""

__all__ = [
    "ZSN2N",
]

from .model import ZSN2N
