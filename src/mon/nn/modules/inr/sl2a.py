#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements FINER network with FINER layers."""

__all__ = [
    "ChebyLayer",
    "LowRankReLULayer",
    "SL2A",
]

import math

import torch

from mon.nn.modules.inr import core


# ----- SL2A's Activation Layer -----
class ChebyKANLayer(torch.nn.Module):
    """This is inspired by Kolmogorov-Arnold Networks but using Chebyshev
    polynomials instead of splines coefficients.

    References:
        https://github.com/SynodicMonth/ChebyKAN
    """

    def __init__(self, in_features: int, out_features: int, degree: int, init_method: str = "xavier_uniform"):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.degree       = degree

        self.cheby_coeffs = torch.nn.Parameter(torch.empty(in_features, out_features, degree + 1))
        # nn.init.normal_(self.cheby_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1)))

        if init_method == "xavier_uniform":
            torch.nn.init.xavier_uniform_(self.cheby_coeffs)
        elif init_method == "kaiming_uniform":
            torch.nn.init.kaiming_uniform_(self.cheby_coeffs, a=0, mode="fan_in", nonlinearity="relu")
        elif init_method == "kaiming_normal":
            torch.nn.init.kaiming_normal_(self.cheby_coeffs, a=0, mode="fan_in", nonlinearity="relu")
        elif init_method == "orthogonal":
            torch.nn.init.orthogonal_(self.cheby_coeffs)
        elif init_method == "uniform":
            torch.nn.init.uniform_(self.cheby_coeffs, a=-0.5, b=0.5)
        elif init_method == "normal":
            torch.nn.init.normal_(self.cheby_coeffs, mean=0.0, std=1 / (in_features * (degree + 1)))

        self.register_buffer("arange", torch.arange(0, degree + 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Since Chebyshev polynomial is defined in [-1, 1]
        # We need to normalize x to [-1, 1] using tanh
        x = torch.tanh(x)
        # View and repeat input degree + 1 times
        x = x.view((-1, self.in_features, 1)).expand(-1, -1, self.degree + 1)  # shape = (batch_size, inputdim, self.degree + 1)
        # Apply acos
        x = x.acos()
        # Multiply by arange [0 .. degree]
        x *= self.arange
        # Apply cos
        x = x.cos()
        # Compute the Chebyshev interpolation
        y = torch.einsum("bid,iod->bo", x, self.cheby_coeffs)  # shape = (batch_size, outdim)
        y = y.view(-1, self.out_features)
        return y


class ChebyLayer(torch.nn.Module):

    def __init__(self, in_features: int, out_features: int, degree: int, init_method: str):
        super().__init__()
        self.cheby = ChebyKANLayer(in_features, out_features, degree, init_method)
        self.norm  = torch.nn.LayerNorm(out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.cheby(x))


class LowRankReLULayer(torch.nn.Module):

    def __init__(
        self,
        in_features     : int,
        out_features    : int,
        rank            : int  = 128,
        bias            : bool = True,
        nonlinear       : str  = "relu",
        linear_init_type: str  = "kaiming_uniform"
    ):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.rank         = rank
        self.nonlinear    = nonlinear

        # Create two smaller matrices for low-rank approximation
        self.weight_left  = torch.nn.Parameter(torch.Tensor(in_features, rank))
        self.weight_right = torch.nn.Parameter(torch.Tensor(rank, out_features))

        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters(linear_init_type)

    def reset_parameters(self, linear_init_type: str = "kaiming_uniform"):
        if linear_init_type == "kaiming_uniform":
            torch.nn.init.kaiming_uniform_(self.weight_left,  a=math.sqrt(5))
            torch.nn.init.kaiming_uniform_(self.weight_right, a=math.sqrt(5))
        elif linear_init_type == "kaiming_normal":
            torch.nn.init.kaiming_normal_(self.weight_left,  a=math.sqrt(5))
            torch.nn.init.kaiming_normal_(self.weight_right, a=math.sqrt(5))
        elif linear_init_type == "orthogonal":
            torch.nn.init.orthogonal_(self.weight_left)
            torch.nn.init.orthogonal_(self.weight_right)
        elif linear_init_type == "uniform":
            torch.nn.init.uniform_(self.weight_left,  a=-0.5, b=0.5)
            torch.nn.init.uniform_(self.weight_right, a=-0.5, b=0.5)
        elif linear_init_type == "normal":
            torch.nn.init.normal_(self.weight_left,  mean=0.0, std=1 / (self.in_features * self.rank))
            torch.nn.init.normal_(self.weight_right, mean=0.0, std=1 / (self.rank * self.out_features))
        elif linear_init_type == "xavier_uniform":
            torch.nn.init.xavier_uniform_(self.weight_left)
            torch.nn.init.xavier_uniform_(self.weight_right)

        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight_left)
            bound     = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute the low-rank approximation of the weight matrix
        weight = torch.matmul(self.weight_left, self.weight_right)
        # Apply the linear transformation
        y = torch.matmul(x, weight)
        if self.bias is not None:
            y += self.bias
        if self.nonlinear == "relu":
            return torch.nn.functional.relu(y)
        else:
            return y


# ----- SL2A -----
class SL2A(torch.nn.Module):

    def __init__(
        self,
        in_features     : int,
        out_features    : int,
        hidden_dim      : int,
        hidden_layers   : int,
        degree          : int  = 256,
        outermost_linear: bool = True,
        nonlinear       : str  = "relu",
        rank            : int  = 32,
        init_method     : str  = "xavier_uniform",
        linear_init_type: str  = "kaiming_uniform"
    ):
        super().__init__()
        self.net = torch.nn.ModuleList()
        self.net.append(ChebyLayer(in_features, hidden_dim, degree=degree, init_method=init_method))

        for i in range(hidden_layers):
            if i == 0:
                self.net.append(LowRankReLULayer(hidden_dim, hidden_dim, rank=rank, nonlinear=nonlinear, linear_init_type=linear_init_type))
            else:
                self.net.append(LowRankReLULayer(hidden_dim, hidden_dim, rank=rank, nonlinear=nonlinear, linear_init_type=linear_init_type))

        if outermost_linear:
            self.net.append(torch.nn.Linear(hidden_dim, out_features))
        else:
            raise NotImplementedError("")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Generates output from image coordinates.

        Args:
            x: Input image tensor as ``torch.Tensor`` for size reference.

        Returns:
            Output tensor as ``torch.Tensor`` from network.
        """
        from mon import vision
        s, _   = vision.image_size(x)
        coords = core.create_coords(s).to(x.device)
        # coords = coords.squeeze()
        for i, layer in enumerate(self.net):
            if i == 0:
                x_ = layer(coords)
                y  = x_
            else:
                y  = layer(torch.einsum("ij,ij->ij", x_, y))
        return y
