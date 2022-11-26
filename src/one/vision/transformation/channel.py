#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Channel transformation.
"""

from __future__ import annotations

import torch.nn.functional as F

from one.core import *


# H1: - Dark Channels ----------------------------------------------------------

def get_dark_channel(image: Tensor, size: int = 15) -> Tensor:
    """
    Get the dark channel prior in the (RGB) image data.
    
    References:
        https://github.com/liboyun/ZID/blob/master/utils/dcp.py
        
    Args:
        image (Tensor): Image of shape [..., 3, H, W] to be transformed,
            where ... means it can have an arbitrary number of leading
            dimensions.
        size (int): Window size. Defaults to 15.
    
    Returns:
        dark_channel (Tensor): Dark channel prior of shape [..., 1, H, W],
            where ... means it can have an arbitrary number of leading
            dimensions.
    """
    assert_tensor_of_ndim(image, 4)
    
    b, c, h, w   = image.shape
    s            = size
    padded       = F.pad(image, (s // 2, s // 2, s // 2, s // 2), "replicate")
    dark_channel = torch.zeros([b, 1, h, w])
    
    for k in range(b):
        for i in range(h):
            for j in range(w):
                dark_channel[k, 0, i, j] = torch.min(padded[k, :, i:(i + s), j:(j + s)])  # CVPR09, eq.5
    return dark_channel


def get_atmosphere_channel(
    image: Tensor,
    p    : float = 0.0001,
    size : int   = 15
) -> Tensor:
    """
    Get the atmosphere light in the (RGB) image data.
    
    References:
        https://github.com/liboyun/ZID/blob/master/utils/dcp.py
        
    Args:
        image (Tensor): Image of shape [..., 3, H, W] to be transformed,
            where ... means it can have an arbitrary number of leading
            dimensions.
        p (float): Percentage of pixels for estimating the atmosphere light.
            Defaults to 0.0001.
        size (int): Window size. Defaults to 15.
        
    Returns:
        atm (Tensor): Atmosphere light ([0, L-1]) for each channel.
    """
    assert_tensor_of_ndim(image, 4)
    b, c, h, w = image.shape
    dark       = get_dark_channel(image=image, size=size)
    flat_i     = torch.reshape(input=image, shape=(b, 3, h * w))
    flat_dark  = torch.ravel(input=dark)
    # Find top h * w * p indexes
    search_idx = torch.argsort(input=(-flat_dark))[:int(h * w * p)]
    # Find the highest intensity for each channel
    atm        = torch.index_select(input=flat_i, dim=-1, index=search_idx)
    atm        = atm.max(dim=-1, keepdim=True)
    return atm[0]
