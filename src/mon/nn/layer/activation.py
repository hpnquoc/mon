#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements activation layers."""

from __future__ import annotations

__all__ = [
    "ArgMax", "CELU", "Clamp", "Clip", "ELU", "FReLU", "GELU", "GLU",
    "Hardshrink", "Hardsigmoid", "Hardswish", "Hardtanh", "LeakyReLU",
    "LogSigmoid", "LogSoftmax", "Mish", "MultiheadAttention", "PReLU", "ReLU",
    "ReLU6", "RReLU", "SELU", "SiLU", "Sigmoid", "SimpleGate", "Softmax",
    "Softmax2d", "Softmin", "Softplus", "Softshrink", "Softsign", "Tanh",
    "Tanhshrink", "Threshold", "hard_sigmoid", "to_act_layer",
]

import functools
import types
from typing import Callable

import torch
from torch import nn
from torch.nn import functional

from mon.core import builtins
from mon.globals import LAYERS
from mon.nn.layer import base


# region Linear Unit

@LAYERS.register()
class CELU(base.PassThroughLayerParsingMixin, nn.CELU):
    pass


@LAYERS.register()
class ELU(base.PassThroughLayerParsingMixin, nn.ELU):
    pass


class FReLU(base.PassThroughLayerParsingMixin, nn.Module):
    
    def __init__(self, c1: int, k: int | list[int] = 3, *args, **kwargs):
        super().__init__()
        k         = builtins.to_2tuple(k)
        self.conv = nn.Conv2d(c1, c1, k, 1, 1, groups=c1)
        self.act  = nn.BatchNorm2d(c1)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        y = max(x, self.act(self.conv(x)))
        return y


@LAYERS.register()
class GELU(base.PassThroughLayerParsingMixin, nn.GELU):
    pass


@LAYERS.register()
class GLU(base.PassThroughLayerParsingMixin, nn.GLU):
    pass


@LAYERS.register()
class LeakyReLU(base.PassThroughLayerParsingMixin, nn.LeakyReLU):
    pass


@LAYERS.register()
class PReLU(base.PassThroughLayerParsingMixin, nn.PReLU):
    pass


@LAYERS.register()
class ReLU(base.PassThroughLayerParsingMixin, nn.ReLU):
    pass


@LAYERS.register()
class ReLU6(base.PassThroughLayerParsingMixin, nn.ReLU6):
    pass


@LAYERS.register()
class RReLU(base.PassThroughLayerParsingMixin, nn.RReLU):
    pass


@LAYERS.register()
class SELU(base.PassThroughLayerParsingMixin, nn.SELU):
    pass


@LAYERS.register()
class SiLU(base.PassThroughLayerParsingMixin, nn.SiLU):
    pass


@LAYERS.register()
class SimpleGate(base.LayerParsingMixin, nn.Module):
    """Simple gate activation unit proposed in the paper: "`Simple Baselines for
    Image Restoration <https://arxiv.org/pdf/2204.04676.pdf>`__".
    """
    
    @classmethod
    def parse_layer_args(cls, f: int, args: list, ch: list) -> tuple[list, list]:
        """Parse layer's arguments :param:`args`, calculate the
        :param:`out_channels`, and update :param:`args`. Also, append the
        :param:`out_channels` to :param:`ch` if needed.

        Args:
            f: From, i.e., the current layer receives output from the f-th layer.
                For example, -1 means from a previous layer; -2 means from 2
                previous layers; [99, 101] means from the 99th and 101st layers.
                This attribute is used in forward pass.
            args: Layer's parameters.
            ch: A :class:`list` containing output channels of previous layers
                (of the model)
        
        Returns:
            The adjusted :param:`args` and :param:`ch`.
        """
        c2 = ch[f] // 2
        ch.append(c2)
        return args, ch
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x      = input
        x1, x2 = x.chunk(chunks=2, dim=1)
        return x1 * x2
    
# endregion


# region Shrink

class Hardshrink(base.PassThroughLayerParsingMixin, nn.Hardshrink):
    pass


class Softshrink(base.PassThroughLayerParsingMixin, nn.Softshrink):
    pass


class Tanhshrink(base.PassThroughLayerParsingMixin, nn.Tanhshrink):
    pass


# endregion


# region Sigmoid

def hard_sigmoid(input: torch.Tensor, inplace: bool = False) -> torch.Tensor:
    if inplace:
        return input.add_(3.0).clamp_(0.0, 6.0).div_(6.0)
    else:
        return functional.relu6(input + 3.0) / 6.0


class Hardsigmoid(base.PassThroughLayerParsingMixin, nn.Hardsigmoid):
    pass


class LogSigmoid(base.PassThroughLayerParsingMixin, nn.LogSigmoid):
    pass


class Sigmoid(base.PassThroughLayerParsingMixin, nn.Sigmoid):
    pass


# endregion


# region Softmax

class LogSoftmax(base.PassThroughLayerParsingMixin, nn.LogSoftmax):
    pass


class Softmax(base.PassThroughLayerParsingMixin, nn.Softmax):
    pass


class Softmax2d(base.PassThroughLayerParsingMixin, nn.Softmax2d):
    pass


# endregion


# region Tanh

class Hardtanh(base.PassThroughLayerParsingMixin, nn.Hardtanh):
    pass


class Tanh(base.PassThroughLayerParsingMixin, nn.Tanh):
    pass


# endregion


# region Misc

class ArgMax(base.PassThroughLayerParsingMixin, nn.Module):
    """Finds indices of maximum values of a tensor along a given dimension
    :param`dim`.
    
    Args:
        dim: A dimension to find indices of maximum values. Default: ``None``.
    """
    
    def __init__(self, dim: int | None = None, *args, **kwargs):
        super().__init__()
        self.dim = dim
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        y = torch.argmax(x, dim=self.dim)
        return y


class Clamp(base.PassThroughLayerParsingMixin, nn.Module):
    """Clamps a tensor' values within a range of :math:`[min, max]`.

    Args:
        min: The lower-bound of the range to be clamped to. Default: ``-1.0``.
        max: The upper-bound of the range to be clamped to. Default: ``-1.0``.
    """
    
    def __init__(
        self,
        min: float = -1.0,
        max: float =  1.0,
        *args, **kwargs
    ):
        super().__init__()
        self.min = min
        self.max = max
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = input
        y = torch.clamp(x, min=self.min, max=self.max)
        return y


Clip = Clamp


class Hardswish(base.PassThroughLayerParsingMixin, nn.Hardswish):
    pass


class Mish(base.PassThroughLayerParsingMixin, nn.Mish):
    pass


class MultiheadAttention(base.PassThroughLayerParsingMixin, nn.MultiheadAttention):
    pass


class Softmin(base.PassThroughLayerParsingMixin, nn.Softmin):
    pass


class Softplus(base.PassThroughLayerParsingMixin, nn.Softplus):
    pass


class Softsign(base.PassThroughLayerParsingMixin, nn.Softsign):
    pass


class Threshold(base.PassThroughLayerParsingMixin, nn.Threshold):
    pass


# endregion


def to_act_layer(act: Callable = ReLU(), *args, **kwargs) -> nn.Module:
    """Create activation layer."""
    # if isinstance(norm, str):
    #     norm = LAYER.build(name=norm)
    act_layer = act
    if act is None or act == False:
        act_layer = nn.Identity()
    elif isinstance(act, Callable | types.FunctionType | functools.partial):
        act_layer = act(*args, **kwargs)
    return act_layer
