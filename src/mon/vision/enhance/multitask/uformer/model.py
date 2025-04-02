#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torch.nn.common_types import _size_2_t


class FastLeFF(nn.Module):

    def __init__(self, dim: int = 32, hidden_dim: int = 128, act_layer: nn.Module = nn.GELU):
        super().__init__()
        # from torch_dwconv import DepthwiseConv2d
        
        self.linear1    = nn.Sequential(nn.Linear(dim, hidden_dim), act_layer())
        # self.dwconv  = nn.Sequential(
        #     DepthwiseConv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
        #     act_layer()
        # )
        self.dwconv     = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, groups=hidden_dim),
            act_layer()
        )
        self.linear2    = nn.Sequential(nn.Linear(hidden_dim, dim))
        self.dim        = dim
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs, hw, c = x.size()  # bs x hw x c
        hh = int(math.sqrt(hw))
        x  = self.linear1(x)
        # Spatial restore
        x  = rearrange(x, " b (h w) (c) -> b c h w ", h=hh, w=hh)  # bs,hidden_dim,32x32
        x  = self.dwconv(x)
        # Flatten
        x  = rearrange(x, " b c h w -> b (h w) c", h=hh, w=hh)
        x  = self.linear2(x)
        return x

    def flops(self, h: int, w: int) -> int:
        flops  = 0
        # fc1
        flops += h * w * self.dim * self.hidden_dim
        # dwconv
        flops += h * w * self.hidden_dim * 3 * 3
        # fc2
        flops += h * w * self.hidden_dim * self.dim
        # print("LeFF:{%.2f}" % (flops / 1e9))
        return flops


def conv(in_channels: int, out_channels: int, kernel_size: _size_2_t, stride: _size_2_t = 1, bias: bool = False):
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=(kernel_size // 2), bias=bias)


# Supervised Attention Module

class SAM(nn.Module):

    def __init__(self, n_feat: int, kernel_size: _size_2_t = 3, bias: bool = True):
        super().__init__()
        self.conv1 = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.conv2 = conv(n_feat, 3, kernel_size, bias=bias)
        self.conv3 = conv(3, n_feat, kernel_size, bias=bias)

    def forward(self, x: torch.Tensor, x_img: torch.Tensor) -> torch.Tensor:
        x1  = self.conv1(x)
        img = self.conv2(x) + x_img
        x2  = torch.sigmoid(self.conv3(img))
        x1  = x1 * x2
        x1  = x1 + x
        return x1, img


#########################################

class ConvBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, strides: _size_2_t = 1):
        super().__init__()
        self.strides      = strides
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.block  = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=strides, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=strides, padding=1),
            nn.LeakyReLU(inplace=True),
        )
        self.conv11 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=strides, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out1 = self.block(x)
        out2 = self.conv11(x)
        out  = out1 + out2
        return out

    def flops(self, h: int, w: int) -> int:
        flops = (h * w * self.in_channels * self.out_channels * (3 * 3 + 1) +
                 h * w * self.out_channels * self.out_channels * 3 * 3)
        return flops


class UNet(nn.Module):

    def __init__(self, block=ConvBlock, dim: int = 32):
        super().__init__()
        self.dim        = dim
        self.ConvBlock1 = ConvBlock(3, dim, strides=1)
        self.pool1      = nn.Conv2d(dim, dim, kernel_size=4, stride=2, padding=1)

        self.ConvBlock2 = block(dim, dim * 2, strides=1)
        self.pool2      = nn.Conv2d(dim * 2, dim * 2, kernel_size=4, stride=2, padding=1)

        self.ConvBlock3 = block(dim * 2, dim * 4, strides=1)
        self.pool3      = nn.Conv2d(dim * 4, dim * 4, kernel_size=4, stride=2, padding=1)

        self.ConvBlock4 = block(dim * 4, dim * 8, strides=1)
        self.pool4      = nn.Conv2d(dim * 8, dim * 8, kernel_size=4, stride=2, padding=1)

        self.ConvBlock5 = block(dim * 8, dim * 16, strides=1)

        self.upv6       = nn.ConvTranspose2d(dim * 16, dim * 8, 2, stride=2)
        self.ConvBlock6 = block(dim * 16, dim * 8, strides=1)

        self.upv7       = nn.ConvTranspose2d(dim * 8, dim * 4, 2, stride=2)
        self.ConvBlock7 = block(dim * 8, dim * 4, strides=1)

        self.upv8       = nn.ConvTranspose2d(dim * 4, dim * 2, 2, stride=2)
        self.ConvBlock8 = block(dim * 4, dim * 2, strides=1)

        self.upv9       = nn.ConvTranspose2d(dim * 2, dim, 2, stride=2)
        self.ConvBlock9 = block(dim * 2, dim, strides=1)

        self.conv10     = nn.Conv2d(dim, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv1  = self.ConvBlock1(x)
        pool1  = self.pool1(conv1)
               
        conv2  = self.ConvBlock2(pool1)
        pool2  = self.pool2(conv2)
               
        conv3  = self.ConvBlock3(pool2)
        pool3  = self.pool3(conv3)
               
        conv4  = self.ConvBlock4(pool3)
        pool4  = self.pool4(conv4)
               
        conv5  = self.ConvBlock5(pool4)
               
        up6    = self.upv6(conv5)
        up6    = torch.cat([up6, conv4], 1)
        conv6  = self.ConvBlock6(up6)
               
        up7    = self.upv7(conv6)
        up7    = torch.cat([up7, conv3], 1)
        conv7  = self.ConvBlock7(up7)
               
        up8    = self.upv8(conv7)
        up8    = torch.cat([up8, conv2], 1)
        conv8  = self.ConvBlock8(up8)
               
        up9    = self.upv9(conv8)
        up9    = torch.cat([up9, conv1], 1)
        conv9  = self.ConvBlock9(up9)

        conv10 = self.conv10(conv9)
        out    = x + conv10

        return out

    def flops(self, h: int, w: int) -> int:
        flops = 0
        flops += self.ConvBlock1.flops(h, w)
        flops += h / 2 * w / 2 * self.dim * self.dim * 4 * 4
        flops += self.ConvBlock2.flops(h / 2, w / 2)
        flops += h / 4 * w / 4 * self.dim * 2 * self.dim * 2 * 4 * 4
        flops += self.ConvBlock3.flops(h / 4, w / 4)
        flops += h / 8 * w / 8 * self.dim * 4 * self.dim * 4 * 4 * 4
        flops += self.ConvBlock4.flops(h / 8, w / 8)
        flops += h / 16 * w / 16 * self.dim * 8 * self.dim * 8 * 4 * 4
        flops += self.ConvBlock5.flops(h / 16, w / 16)
        flops += h / 8 * w / 8 * self.dim * 16 * self.dim * 8 * 2 * 2
        flops += self.ConvBlock6.flops(h / 8, w / 8)
        flops += h / 4 * w / 4 * self.dim * 8 * self.dim * 4 * 2 * 2
        flops += self.ConvBlock7.flops(h / 4, w / 4)
        flops += h / 2 * w / 2 * self.dim * 4 * self.dim * 2 * 2 * 2
        flops += self.ConvBlock8.flops(h / 2, w / 2)
        flops += h * w * self.dim * 2 * self.dim * 2 * 2
        flops += self.ConvBlock9.flops(h, w)
        flops += h * w * self.dim * 3 * 3 * 3
        return flops


class LPU(nn.Module):
    """Local Perception Unit to extract local information.
    LPU(X) = DWConv(X) + X
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.depthwise    = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=True)
        self.in_channels  = in_channels
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l, c = x.shape
        h       = int(math.sqrt(l))
        w       = int(math.sqrt(l))
        x       = x.transpose(1, 2).contiguous().view(b, c, h, w)
        result  = (self.depthwise(x) + x).flatten(2).transpose(1, 2).contiguous()  # b h*w c
        return result

    def flops(self, h: int, w: int) -> int:
        flops  = 0
        # conv
        flops += h * w * self.out_channels * 3 * 3
        return flops


#########################################

class PosCNN(nn.Module):

    def __init__(self, in_channels: int, embed_dim: int = 768, s: int = 1):
        super().__init__()
        self.proj = nn.Sequential(nn.Conv2d(in_channels, embed_dim, 3, s, 1, bias=True, groups=embed_dim))
        self.s    = s

    def forward(self, x: torch.Tensor, h: int = None, w: int = None):
        b, n, c    = x.shape
        h          = h or int(math.sqrt(n))
        w          = w or int(math.sqrt(n))
        feat_token = x
        cnn_feat   = feat_token.transpose(1, 2).view(b, c, h, w)
        if self.s == 1:
            x = self.proj(cnn_feat) + cnn_feat
        else:
            x = self.proj(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        return x

    def no_weight_decay(self):
        return ["proj.%d.weight" % i for i in range(4)]


class SELayer(nn.Module):

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc       = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        self.reduction = reduction

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B, N, C]
        x = torch.transpose(x, 1, 2)  # [B, C, N]
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        x = x * y.expand_as(x)
        x = torch.transpose(x, 1, 2)  # [B, N, C]
        return x

    def flops(self) -> int:
        flops  = 0
        flops += self.channel * self.channel / self.reduction * 2
        return flops


class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channels: Number of channels of the input feature map
        kernel_size: Adaptive selection of kernel size
    """

    def __init__(self, channels: int, kernel_size: _size_2_t = 3):
        super().__init__()
        self.avg_pool    = nn.AdaptiveAvgPool2d(1)
        self.conv        = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid     = nn.Sigmoid()
        self.channel     = channels
        self.kernel_size = kernel_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)
        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # Multi-scale information fusion
        y = self.sigmoid(y)
        return x * y.expand_as(x)

    def flops(self) -> int:
        flops  = 0
        flops += self.channel * self.channel * self.kernel_size
        return flops


class eca_layer_1d(nn.Module):
    """Constructs a ECA module.
    Args:
        channels: Number of channels of the input feature map
        kernel_size: Adaptive selection of kernel size
    """

    def __init__(self, channels: int, kernel_size: _size_2_t = 3):
        super().__init__()
        self.avg_pool    = nn.AdaptiveAvgPool1d(1)
        self.conv        = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid     = nn.Sigmoid()
        self.channel     = channels
        self.kernel_size = kernel_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # b hw c
        # feature descriptor on the global spatial information
        y = self.avg_pool(x.transpose(-1, -2))
        # Two different branches of ECA module
        y = self.conv(y.transpose(-1, -2))
        # Multi-scale information fusion
        y = self.sigmoid(y)
        return x * y.expand_as(x)

    def flops(self) -> int:
        flops  = 0
        flops += self.channel * self.channel * self.kernel_size
        return flops


class SepConv2d(nn.Module):

    def __init__(
        self,
        in_channels : int,
        out_channels: int,
        kernel_size : _size_2_t,
        stride      : _size_2_t = 1,
        padding     : _size_2_t = 0,
        dilation    : int       = 1,
        act_layer   : nn.Module = nn.ReLU
    ):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels  = in_channels,
            out_channels = in_channels,
            kernel_size  = kernel_size,
            stride       = stride,
            padding      = padding,
            dilation     = dilation,
            groups       = in_channels
        )
        self.pointwise    = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.act_layer    = act_layer() if act_layer is not None else nn.Identity()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.kernel_size  = kernel_size
        self.stride       = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.act_layer(x)
        x = self.pointwise(x)
        return x

    def flops(self, hw: int) -> int:
        flops  = 0
        flops += hw * self.in_channels * self.kernel_size ** 2 / self.stride ** 2
        flops += hw * self.in_channels * self.out_channels
        print("SeqConv2d:{%.2f}" % (flops / 1e9))
        return flops


######## Embedding for q,k,v ########

class ConvProjection(nn.Module):
    
    def __init__(
        self,
        dim        : int,
        heads      : int       = 8,
        dim_head   : int       = 64,
        kernel_size: _size_2_t = 3,
        q_stride   : int       = 1,
        k_stride   : int       = 1,
        v_stride   : int       = 1,
        dropout    : float     = 0.0,
        last_stage : bool      = False,
        bias       : bool      = True
    ):
        super().__init__()
        inner_dim  = dim_head * heads
        self.heads = heads
        pad        = (kernel_size - q_stride) // 2
        self.to_q  = SepConv2d(dim, inner_dim, kernel_size, q_stride, pad, bias)
        self.to_k  = SepConv2d(dim, inner_dim, kernel_size, k_stride, pad, bias)
        self.to_v  = SepConv2d(dim, inner_dim, kernel_size, v_stride, pad, bias)

    def forward(self, x: torch.Tensor, attn_kv: torch.Tensor = None) -> torch.Tensor:
        b, n, c, h = *x.shape, self.heads
        l = int(math.sqrt(n))
        w = int(math.sqrt(n))

        attn_kv = x if attn_kv is None else attn_kv
        x       = rearrange(x, "b (l w) c -> b c l w", l=l, w=w)
        attn_kv = rearrange(attn_kv, "b (l w) c -> b c l w", l=l, w=w)
        
        q = self.to_q(x)
        q = rearrange(q, "b (h d) l w -> b h (l w) d", h=h)

        k = self.to_k(attn_kv)
        v = self.to_v(attn_kv)
        k = rearrange(k, "b (h d) l w -> b h (l w) d", h=h)
        v = rearrange(v, "b (h d) l w -> b h (l w) d", h=h)
        return q, k, v

    def flops(self, q_L, kv_L=None):
        kv_L   = kv_L or q_L
        flops  = 0
        flops += self.to_q.flops(q_L)
        flops += self.to_k.flops(kv_L)
        flops += self.to_v.flops(kv_L)
        return flops


class LinearProjection(nn.Module):
    
    def __init__(
        self,
        dim     : int,
        heads   : int   = 8,
        dim_head: int   = 64,
        dropout : float = 0.0,
        bias    : bool  = True
    ):
        super().__init__()
        inner_dim      = dim_head * heads
        self.heads     = heads
        self.to_q      = nn.Linear(dim, inner_dim,     bias=bias)
        self.to_kv     = nn.Linear(dim, inner_dim * 2, bias=bias)
        self.dim       = dim
        self.inner_dim = inner_dim

    def forward(self, x: torch.Tensor, attn_kv: torch.Tensor = None) -> torch.Tensor:
        b_, n, c = x.shape
        if attn_kv is not None:
            attn_kv = attn_kv.unsqueeze(0).repeat(b_, 1, 1)
        else:
            attn_kv = x
        n_kv = attn_kv.size(1)
        q    = self.to_q(x).reshape(b_, n, 1, self.heads, c // self.heads).permute(2, 0, 3, 1, 4)
        kv   = self.to_kv(attn_kv).reshape(b_, n_kv, 2, self.heads, c // self.heads).permute(2, 0, 3, 1, 4)
        q    = q[0]
        k, v = kv[0], kv[1]
        return q, k, v

    def flops(self, q_L, kv_L=None) -> int:
        kv_L  = kv_L or q_L
        flops = q_L * self.dim * self.inner_dim + kv_L * self.dim * self.inner_dim * 2
        return flops

########### window-based self-attention #############

class WindowAttention(nn.Module):
    def __init__(
        self,
        dim             : int,
        win_size        : int,
        num_heads       : int,
        token_projection: str   = "linear",
        qkv_bias        : bool  = True,
        qk_scale        : float = None,
        attn_drop       : float = 0.0,
        proj_drop       : float = 0.0
    ):
        super().__init__()
        self.dim       = dim
        self.win_size  = win_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim       = dim // num_heads
        self.scale     = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * win_size[0] - 1) * (2 * win_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.win_size[0])  # [0,...,Wh-1]
        coords_w = torch.arange(self.win_size[1])  # [0,...,Ww-1]
        coords   = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten  = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2,
                                                  0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.win_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.win_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.win_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)
        trunc_normal_(self.relative_position_bias_table, std=.02)

        if token_projection == "conv":
            self.qkv = ConvProjection(dim, num_heads, dim // num_heads, bias=qkv_bias)
        elif token_projection == "linear":
            self.qkv = LinearProjection(dim, num_heads, dim // num_heads, bias=qkv_bias)
        else:
            raise Exception("Projection error!")

        self.token_projection = token_projection
        self.attn_drop        = nn.Dropout(attn_drop)
        self.proj             = nn.Linear(dim, dim)
        self.proj_drop        = nn.Dropout(proj_drop)

        self.softmax          = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, attn_kv: torch.Tensor = None, mask: torch.Tensor = None) -> torch.Tensor:
        b_, n, c = x.shape
        q, k, v  = self.qkv(x, attn_kv)
        q        = q * self.scale
        attn     = (q @ k.transpose(-2, -1))
        
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.win_size[0] * self.win_size[1],
            self.win_size[0] * self.win_size[1],
            -1
        )  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        ratio                  = attn.size(-1) // relative_position_bias.size(-1)
        relative_position_bias = repeat(relative_position_bias, "nH l c -> nH l (c d)", d=ratio)

        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW   = mask.shape[0]
            mask = repeat(mask, "nW m n -> nW m (n d)", d=ratio)
            attn = attn.view(b_ // nW, nW, self.num_heads, n, n * ratio) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n * ratio)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, win_size={self.win_size}, num_heads={self.num_heads}"

    def flops(self, h: int, w: int) -> int:
        # calculate flops for 1 window with token length of N
        # print(N, self.dim)
        flops  = 0
        N      = self.win_size[0] * self.win_size[1]
        nW     = h * w / N
        # qkv = self.qkv(x)
        # flops += N * self.dim * 3 * self.dim
        flops += self.qkv.flops(h * w, h * w)
        # attn = (q @ k.transpose(-2, -1))
        flops += nW * self.num_heads * N * (self.dim // self.num_heads) * N
        # x = (attn @ v)
        flops += nW * self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += nW * N * self.dim * self.dim
        print("W-MSA:{%.2f}" % (flops / 1e9))
        return flops


########### Self-attention #############

class Attention(nn.Module):
    
    def __init__(
        self,
        dim             : int,
        num_heads       : int,
        token_projection: str   = "linear",
        qkv_bias        : bool  = True,
        qk_scale        : float = None,
        attn_drop       : float = 0.0,
        proj_drop       : float = 0.0
    ):
        super().__init__()
        self.dim       = dim
        self.num_heads = num_heads
        head_dim       = dim // num_heads
        self.scale     = qk_scale or head_dim ** -0.5
        self.qkv = LinearProjection(dim, num_heads, dim // num_heads, bias=qkv_bias)
        self.token_projection = token_projection
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj      = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax   = nn.Softmax(dim=-1)

    def forward(
        self,
        x      : torch.Tensor,
        attn_kv: torch.Tensor = None,
        mask   : torch.Tensor = None
    ) -> torch.Tensor:
        b_, n, c = x.shape
        q, k, v  = self.qkv(x, attn_kv)
        q        = q * self.scale
        attn     = (q @ k.transpose(-2, -1))
        
        # relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
        #     self.win_size[0] * self.win_size[1], self.win_size[0] * self.win_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        # relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        # ratio = attn.size(-1)//relative_position_bias.size(-1)
        # relative_position_bias = repeat(relative_position_bias, 'nH l c -> nH l (c d)', d = ratio)

        # attn = attn + relative_position_bias.unsqueeze(0)
        
        if mask is not None:
            nW = mask.shape[0]
            # mask = repeat(mask, 'nW m n -> nW m (n d)',d = ratio)
            attn = attn.view(b_ // nW, nW, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, num_heads={self.num_heads}"

    def flops(self, q_num, kv_num) -> int:
        # calculate flops for 1 window with token length of N
        # print(N, self.dim)
        flops = 0
        # N = self.win_size[0]*self.win_size[1]
        # nW = H*W/N
        # qkv = self.qkv(x)
        # flops += N * self.dim * 3 * self.dim
        flops += self.qkv.flops(q_num, kv_num)
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * q_num * (self.dim // self.num_heads) * kv_num
        #  x = (attn @ v)
        flops += self.num_heads * q_num * (self.dim // self.num_heads) * kv_num
        # x = self.proj(x)
        flops += q_num * self.dim * self.dim
        print("MCA:{%.2f}" % (flops / 1e9))
        return flops


########### Feed-forward network #############

class Mlp(nn.Module):
    
    def __init__(
        self,
        in_features    : int,
        hidden_features: int   = None,
        out_features   : int   = None,
        act_layer              = nn.GELU,
        drop           : float = 0.0
    ):
        super().__init__()
        out_features         = out_features    or in_features
        hidden_features      = hidden_features or in_features
        self.fc1             = nn.Linear(in_features, hidden_features)
        self.act             = act_layer()
        self.fc2             = nn.Linear(hidden_features, out_features)
        self.drop            = nn.Dropout(drop)
        self.in_features     = in_features
        self.hidden_features = hidden_features
        self.out_features    = out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

    def flops(self, h: int, w: int) -> int:
        flops  = 0
        # fc1
        flops += h * w * self.in_features * self.hidden_features
        # fc2
        flops += h * w * self.hidden_features * self.out_features
        print("MLP:{%.2f}" % (flops / 1e9))
        return flops


class LeFF(nn.Module):
    
    def __init__(
        self,
        dim       : int   = 32,
        hidden_dim: int   = 128,
        act_layer         = nn.GELU,
        drop      : float = 0.0,
        use_eca   : bool  = False
    ):
        super().__init__()
        self.linear1 = nn.Sequential(nn.Linear(dim, hidden_dim), act_layer())
        self.dwconv  = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, groups=hidden_dim, kernel_size=3, stride=1, padding=1),
            act_layer()
        )
        self.linear2    = nn.Sequential(nn.Linear(hidden_dim, dim))
        self.dim        = dim
        self.hidden_dim = hidden_dim
        self.eca        = eca_layer_1d(dim) if use_eca else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # bs x hw x c
        bs, hw, c = x.size()
        hh = int(math.sqrt(hw))
        x  = self.linear1(x)
        # Spatial restore
        x = rearrange(x, ' b (h w) (c) -> b c h w ', h=hh, w=hh)
        # bs,hidden_dim,32x32
        x = self.dwconv(x)
        # Flatten
        x = rearrange(x, ' b c h w -> b (h w) c', h=hh, w=hh)
        x = self.linear2(x)
        x = self.eca(x)
        return x

    def flops(self, h: int, w: int) -> int:
        flops = 0
        # fc1
        flops += h * w * self.dim * self.hidden_dim
        # dwconv
        flops += h * w * self.hidden_dim * 3 * 3
        # fc2
        flops += h * w * self.hidden_dim * self.dim
        print("LeFF:{%.2f}" % (flops / 1e9))
        # eca 
        if hasattr(self.eca, "flops"):
            flops += self.eca.flops()
        return flops


########### Window operation #############

def window_partition(x, win_size, dilation_rate=1):
    b, h, w, c = x.shape
    if dilation_rate != 1:
        x = x.permute(0, 3, 1, 2)  # b, c, h, w
        assert type(dilation_rate) is int, "dilation_rate should be a int"
        x = F.unfold(
            x,
            kernel_size = win_size,
            dilation    = dilation_rate,
            padding     = 4 * (dilation_rate - 1),
            stride      = win_size
        )  # B, C*Wh*Ww, H/Wh*W/Ww
        windows = x.permute(0, 2, 1).contiguous().view(-1, c, win_size, win_size)  # B' ,C ,Wh ,Ww
        windows = windows.permute(0, 2, 3, 1).contiguous()  # B' ,Wh ,Ww ,C
    else:
        x       = x.view(b, h // win_size, win_size, w // win_size, win_size, c)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, win_size, win_size, c)  # B' ,Wh ,Ww ,C
    return windows


def window_reverse(windows, win_size, h, w, dilation_rate=1):
    # B' ,Wh ,Ww ,C
    b = int(windows.shape[0] / (h * w / win_size / win_size))
    x = windows.view(b, h // win_size, w // win_size, win_size, win_size, -1)
    if dilation_rate != 1:
        x = windows.permute(0, 5, 3, 4, 1, 2).contiguous()  # B, C*Wh*Ww, H/Wh*W/Ww
        x = F.fold(
            x, (h, w),
            kernel_size = win_size,
            dilation    = dilation_rate,
            padding     = 4 * (dilation_rate - 1),
            stride      = win_size
        )
    else:
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x


# Downsample Block

class Downsample(nn.Module):
    
    def __init__(self, in_channel: int, out_channel: int):
        super(Downsample, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=4, stride=2, padding=1),
        )
        self.in_channel  = in_channel
        self.out_channel = out_channel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l, c = x.shape
        # import pdb;pdb.set_trace()
        h   = int(math.sqrt(l))
        w   = int(math.sqrt(l))
        x   = x.transpose(1, 2).contiguous().view(b, c, h, w)
        out = self.conv(x).flatten(2).transpose(1, 2).contiguous()  # B H*W C
        return out

    def flops(self, h: int, w: int):
        flops = 0
        # conv
        flops += h / 2 * w / 2 * self.in_channel * self.out_channel * 4 * 4
        print("Downsample:{%.2f}" % (flops / 1e9))
        return flops


# Upsample Block

class Upsample(nn.Module):
    
    def __init__(self, in_channel: int, out_channel: int):
        super(Upsample, self).__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2),
        )
        self.in_channel  = in_channel
        self.out_channel = out_channel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l, c = x.shape
        h   = int(math.sqrt(l))
        w   = int(math.sqrt(l))
        x   = x.transpose(1, 2).contiguous().view(b, c, h, w)
        out = self.deconv(x).flatten(2).transpose(1, 2).contiguous()  # B H*W C
        return out

    def flops(self, h: int, w: int):
        flops = 0
        # conv
        flops += h * 2 * w * 2 * self.in_channel * self.out_channel * 2 * 2
        print("Upsample:{%.2f}" % (flops / 1e9))
        return flops


# Input Projection
class InputProj(nn.Module):
    
    def __init__(self, in_channel=3, out_channel=64, kernel_size=3, stride=1, norm_layer=None, act_layer=nn.LeakyReLU):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=kernel_size // 2),
            act_layer(inplace=True)
        )
        if norm_layer is not None:
            self.norm = norm_layer(out_channel)
        else:
            self.norm = None
        self.in_channel  = in_channel
        self.out_channel = out_channel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2).contiguous()  # B H*W C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self, h: int, w: int):
        flops  = 0
        # conv
        flops += h * w * self.in_channel * self.out_channel * 3 * 3
        if self.norm is not None:
            flops += h * w * self.out_channel
        print("Input_proj:{%.2f}" % (flops / 1e9))
        return flops


# Output Projection
class OutputProj(nn.Module):
    
    def __init__(self, in_channel=64, out_channel=3, kernel_size=3, stride=1, norm_layer=None, act_layer=None):
        super().__init__()
        if act_layer is not None:
            self.proj = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=kernel_size // 2),
                act_layer(inplace=True),
            )
        else:
            self.proj = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=kernel_size // 2),
            )
        if norm_layer is not None:
            self.norm = norm_layer(out_channel)
        else:
            self.norm = None
        self.in_channel  = in_channel
        self.out_channel = out_channel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l, c = x.shape
        h = int(math.sqrt(l))
        w = int(math.sqrt(l))
        x = x.transpose(1, 2).view(b, c, h, w)
        x = self.proj(x)
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self, h: int, w: int):
        flops = 0
        # conv
        flops += h * w * self.in_channel * self.out_channel * 3 * 3
        if self.norm is not None:
            flops += h * w * self.out_channel
        print("Output_proj:{%.2f}" % (flops / 1e9))
        return flops


# LeWinTransformer

class LeWinTransformerBlock(nn.Module):
    
    def __init__(
        self, dim, input_resolution, num_heads,
        win_size         = 8,
        shift_size       = 0,
        mlp_ratio        = 4.0,
        qkv_bias         = True,
        qk_scale         = None,
        drop             = 0.0,
        attn_drop        = 0.0,
        drop_path        = 0.0,
        act_layer        = nn.GELU,
        norm_layer       = nn.LayerNorm,
        token_projection = "linear",
        token_mlp        = "leff",
        modulator        = False,
        cross_modulator  = False
    ):
        super().__init__()
        self.dim              = dim
        self.input_resolution = input_resolution
        self.num_heads        = num_heads
        self.win_size         = win_size
        self.shift_size       = shift_size
        self.mlp_ratio        = mlp_ratio
        self.token_mlp        = token_mlp
        if min(self.input_resolution) <= self.win_size:
            self.shift_size = 0
            self.win_size   = min(self.input_resolution)
        assert 0 <= self.shift_size < self.win_size, "shift_size must in 0-win_size"

        if modulator:
            self.modulator = nn.Embedding(win_size * win_size, dim)  # modulator
        else:
            self.modulator = None

        if cross_modulator:
            self.cross_modulator = nn.Embedding(win_size * win_size, dim)  # cross_modulator
            self.cross_attn = Attention(
                dim, num_heads,
                qkv_bias         = qkv_bias,
                qk_scale         = qk_scale,
                attn_drop        = attn_drop,
                proj_drop        = drop,
                token_projection = token_projection,
            )
            self.norm_cross = norm_layer(dim)
        else:
            self.cross_modulator = None

        self.norm1 = norm_layer(dim)
        self.attn  = WindowAttention(
            dim,
            win_size         = to_2tuple(self.win_size),
            num_heads        = num_heads,
            qkv_bias         = qkv_bias,
            qk_scale         = qk_scale,
            attn_drop        = attn_drop,
            proj_drop        = drop,
            token_projection = token_projection
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2     = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        if token_mlp in ["ffn", "mlp"]:
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        elif token_mlp == "leff":
            self.mlp = LeFF(dim, mlp_hidden_dim, act_layer=act_layer, drop=drop)
        elif token_mlp == "fastleff":
            self.mlp = FastLeFF(dim, mlp_hidden_dim, act_layer=act_layer)
        else:
            raise Exception("FFN error!")

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"win_size={self.win_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio},modulator={self.modulator}"

    def forward(self, x, mask=None):
        b, l, c = x.shape
        h = int(math.sqrt(l))
        w = int(math.sqrt(l))

        # input mask
        if mask is None:
            input_mask = F.interpolate(mask, size=(h, w)).permute(0, 2, 3, 1)
            input_mask_windows = window_partition(input_mask, self.win_size)  # nW, win_size, win_size, 1
            attn_mask = input_mask_windows.view(-1, self.win_size * self.win_size)  # nW, win_size*win_size
            attn_mask = attn_mask.unsqueeze(2) * attn_mask.unsqueeze(1)  # nW, win_size*win_size, win_size*win_size
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        ##shift mask
        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            shift_mask = torch.zeros((1, h, w, 1)).type_as(x)
            h_slices = (slice(0, -self.win_size),
                        slice(-self.win_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.win_size),
                        slice(-self.win_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    shift_mask[:, h, w, :] = cnt
                    cnt += 1
            shift_mask_windows = window_partition(shift_mask, self.win_size)  # nW, win_size, win_size, 1
            shift_mask_windows = shift_mask_windows.view(-1, self.win_size * self.win_size)  # nW, win_size*win_size
            shift_attn_mask    = shift_mask_windows.unsqueeze(1) - shift_mask_windows.unsqueeze(2)  # nW, win_size*win_size, win_size*win_size
            shift_attn_mask    = shift_attn_mask.masked_fill(shift_attn_mask != 0, float(-100.0)).masked_fill(shift_attn_mask == 0, float(0.0))
            attn_mask          = attn_mask + shift_attn_mask if attn_mask is not None else shift_attn_mask

        if self.cross_modulator is not None:
            shortcut = x
            x_cross  = self.norm_cross(x)
            x_cross  = self.cross_attn(x, self.cross_modulator.weight)
            x = shortcut + x_cross

        shortcut = x
        x = self.norm1(x)
        x = x.view(b, h, w, c)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.win_size)  # nW*B, win_size, win_size, C  N*C->C
        x_windows = x_windows.view(-1, self.win_size * self.win_size, c)  # nW*B, win_size*win_size, C

        # with_modulator
        if self.modulator is not None:
            wmsa_in = self.with_pos_embed(x_windows, self.modulator.weight)
        else:
            wmsa_in = x_windows

        # W-MSA/SW-MSA
        attn_windows = self.attn(wmsa_in, mask=attn_mask)  # nW*B, win_size*win_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.win_size, self.win_size, c)
        shifted_x    = window_reverse(attn_windows, self.win_size, h, w)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(b, h * w, c)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        del attn_mask
        return x

    def flops(self):
        flops = 0
        h, w  = self.input_resolution
        if self.cross_modulator is not None:
            flops += self.dim * h * w
            flops += self.cross_attn.flops(h * w, self.win_size * self.win_size)
        # norm1
        flops += self.dim * h * w
        # W-MSA/SW-MSA
        flops += self.attn.flops(h, w)
        # norm2
        flops += self.dim * h * w
        # mlp
        flops += self.mlp.flops(h, w)
        # print("LeWin:{%.2f}"%(flops/1e9))
        return flops


# Basic layer of Uformer

class BasicUformerLayer(nn.Module):
    
    def __init__(
        self, dim, output_dim, input_resolution, depth, num_heads, win_size,
        mlp_ratio        = 4.0,
        qkv_bias         = True,
        qk_scale         = None,
        drop             = 0.0,
        attn_drop        = 0.0,
        drop_path        = 0.0,
        norm_layer       = nn.LayerNorm,
        use_checkpoint   = False,
        token_projection = "linear",
        token_mlp        = "ffn",
        shift_flag       = True,
        modulator        = False,
        cross_modulator  = False
    ):

        super().__init__()
        self.dim              = dim
        self.input_resolution = input_resolution
        self.depth            = depth
        self.use_checkpoint   = use_checkpoint
        # build blocks
        if shift_flag:
            self.blocks = nn.ModuleList([
                LeWinTransformerBlock(
                    dim              = dim,
                    input_resolution = input_resolution,
                    num_heads        = num_heads,
                    win_size         = win_size,
                    shift_size       = 0 if (i % 2 == 0) else win_size // 2,
                    mlp_ratio        = mlp_ratio,
                    qkv_bias         = qkv_bias,
                    qk_scale         = qk_scale,
                    drop             = drop,
                    attn_drop        = attn_drop,
                    drop_path        = drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer       = norm_layer,
                    token_projection = token_projection,
                    token_mlp        = token_mlp,
                    modulator        = modulator,
                    cross_modulator  = cross_modulator
                )
                for i in range(depth)])
        else:
            self.blocks = nn.ModuleList([
                LeWinTransformerBlock(
                    dim              = dim,
                    input_resolution = input_resolution,
                    num_heads        = num_heads,
                    win_size         = win_size,
                    shift_size       = 0,
                    mlp_ratio        = mlp_ratio,
                    qkv_bias         = qkv_bias,
                    qk_scale         = qk_scale,
                    drop             = drop,
                    attn_drop        = attn_drop,
                    drop_path        = drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer       = norm_layer,
                    token_projection = token_projection,
                    token_mlp        = token_mlp,
                    modulator        = modulator,
                    cross_modulator  = cross_modulator
                )
                for i in range(depth)])

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x, mask)
        return x

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        return flops


class Uformer(nn.Module):
    
    def __init__(
        self,
        img_size        : int       = 256,
        in_chans        : int       = 3,
        dd_in           : int       = 3,
        embed_dim       : int       = 32,
        depths          : list      = [2, 2, 2, 2, 2 , 2 , 2, 2, 2],
        num_heads       : list      = [1, 2, 4, 8, 16, 16, 8, 4, 2],
        win_size        : int       = 8,
        mlp_ratio       : float     = 4.0,
        qkv_bias        : bool      = True,
        qk_scale        : float     = None,
        drop_rate       : float     = 0.0,
        attn_drop_rate  : float     = 0.0,
        drop_path_rate  : float     = 0.1,
        norm_layer      : nn.Module = nn.LayerNorm,
        patch_norm      : bool      = True,
        use_checkpoint  : bool      = False,
        token_projection: str       = "linear",
        token_mlp       : str       = "leff",
        dowsample       : nn.Module = Downsample,
        upsample        : nn.Module = Upsample,
        shift_flag      : bool      = True,
        modulator       : bool      = False,
        cross_modulator : bool      = False,
        **kwargs
    ):
        super().__init__()
        self.num_enc_layers   = len(depths) // 2
        self.num_dec_layers   = len(depths) // 2
        self.embed_dim        = embed_dim
        self.patch_norm       = patch_norm
        self.mlp_ratio        = mlp_ratio
        self.token_projection = token_projection
        self.mlp              = token_mlp
        self.win_size         = win_size
        self.reso             = img_size
        self.pos_drop         = nn.Dropout(p=drop_rate)
        self.dd_in            = dd_in

        # stochastic depth
        enc_dpr  = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths[:self.num_enc_layers]))]
        conv_dpr = [drop_path_rate] * depths[4]
        dec_dpr  = enc_dpr[::-1]

        # build layers

        # Input/Output
        self.input_proj  =  InputProj(in_channel=dd_in,        out_channel=embed_dim, kernel_size=3, stride=1, act_layer=nn.LeakyReLU)
        self.output_proj = OutputProj(in_channel=2 * embed_dim, out_channel=in_chans, kernel_size=3, stride=1)

        # Encoder
        self.encoderlayer_0 = BasicUformerLayer(
            dim              = embed_dim,
            output_dim       = embed_dim,
            input_resolution = (img_size, img_size),
            depth            = depths[0],
            num_heads        = num_heads[0],
            win_size         = win_size,
            mlp_ratio        = self.mlp_ratio,
            qkv_bias         = qkv_bias,
            qk_scale         = qk_scale,
            drop             = drop_rate,
            attn_drop        = attn_drop_rate,
            drop_path        = enc_dpr[sum(depths[: 0]) : sum(depths[: 1])],
            norm_layer       = norm_layer,
            use_checkpoint   = use_checkpoint,
            token_projection = token_projection,
            token_mlp        = token_mlp,
            shift_flag       = shift_flag
        )
        self.dowsample_0    = dowsample(embed_dim, embed_dim * 2)
        self.encoderlayer_1 = BasicUformerLayer(
            dim              = embed_dim * 2,
            output_dim       = embed_dim * 2,
            input_resolution = (img_size // 2, img_size // 2),
            depth            = depths[1],
            num_heads        = num_heads[1],
            win_size         = win_size,
            mlp_ratio        = self.mlp_ratio,
            qkv_bias         = qkv_bias,
            qk_scale         = qk_scale,
            drop             = drop_rate,
            attn_drop        = attn_drop_rate,
            drop_path        = enc_dpr[sum(depths[: 1]): sum(depths[: 2])],
            norm_layer       = norm_layer,
            use_checkpoint   = use_checkpoint,
            token_projection = token_projection,
            token_mlp        = token_mlp,
            shift_flag       = shift_flag
        )
        self.dowsample_1    = dowsample(embed_dim * 2, embed_dim * 4)
        self.encoderlayer_2 = BasicUformerLayer(
            dim              = embed_dim * 4,
            output_dim       = embed_dim * 4,
            input_resolution = (img_size // (2 ** 2), img_size // (2 ** 2)),
            depth            = depths[2],
            num_heads        = num_heads[2],
            win_size         = win_size,
            mlp_ratio        = self.mlp_ratio,
            qkv_bias         = qkv_bias,
            qk_scale         = qk_scale,
            drop             = drop_rate,
            attn_drop        = attn_drop_rate,
            drop_path        = enc_dpr[sum(depths[: 2]): sum(depths[: 3])],
            norm_layer       = norm_layer,
            use_checkpoint   = use_checkpoint,
            token_projection = token_projection,
            token_mlp        = token_mlp,
            shift_flag       = shift_flag
        )
        self.dowsample_2    = dowsample(embed_dim * 4, embed_dim * 8)
        self.encoderlayer_3 = BasicUformerLayer(
            dim              = embed_dim * 8,
            output_dim       = embed_dim * 8,
            input_resolution = (img_size // (2 ** 3), img_size // (2 ** 3)),
            depth            = depths[3],
            num_heads        = num_heads[3],
            win_size         = win_size,
            mlp_ratio        = self.mlp_ratio,
            qkv_bias         = qkv_bias,
            qk_scale         = qk_scale,
            drop             = drop_rate,
            attn_drop        = attn_drop_rate,
            drop_path        = enc_dpr[sum(depths[: 3]): sum(depths[: 4])],
            norm_layer       = norm_layer,
            use_checkpoint   = use_checkpoint,
            token_projection = token_projection,
            token_mlp        = token_mlp,
            shift_flag       = shift_flag
        )
        self.dowsample_3 = dowsample(embed_dim * 8, embed_dim * 16)

        # Bottleneck
        self.conv = BasicUformerLayer(
            dim              = embed_dim * 16,
            output_dim       = embed_dim * 16,
            input_resolution = (img_size // (2 ** 4), img_size // (2 ** 4)),
            depth            = depths[4],
            num_heads        = num_heads[4],
            win_size         = win_size,
            mlp_ratio        = self.mlp_ratio,
            qkv_bias         = qkv_bias,
            qk_scale         = qk_scale,
            drop             = drop_rate ,
            attn_drop        = attn_drop_rate,
            drop_path        = conv_dpr,
            norm_layer       = norm_layer,
            use_checkpoint   = use_checkpoint,
            token_projection = token_projection,
            token_mlp        = token_mlp,
            shift_flag       = shift_flag
        )

        # Decoder
        self.upsample_0     = upsample(embed_dim * 16, embed_dim * 8)
        self.decoderlayer_0 = BasicUformerLayer(
            dim              = embed_dim * 16,
            output_dim       = embed_dim * 16,
            input_resolution = (img_size // (2 ** 3), img_size // (2 ** 3)),
            depth            = depths[5],
            num_heads        = num_heads[5],
            win_size         = win_size,
            mlp_ratio        = self.mlp_ratio,
            qkv_bias         = qkv_bias,
            qk_scale         = qk_scale,
            drop             = drop_rate,
            attn_drop        = attn_drop_rate,
            drop_path        = dec_dpr[        : depths[5]],
            norm_layer       = norm_layer,
            use_checkpoint   = use_checkpoint,
            token_projection = token_projection,
            token_mlp        = token_mlp,
            shift_flag       = shift_flag,
            modulator        = modulator,
            cross_modulator  = cross_modulator
        )
        self.upsample_1 = upsample(embed_dim * 16, embed_dim * 4)
        self.decoderlayer_1 = BasicUformerLayer(
            dim              = embed_dim * 8,
            output_dim       = embed_dim * 8,
            input_resolution = (img_size // (2 ** 2), img_size // (2 ** 2)),
            depth            = depths[6],
            num_heads        = num_heads[6],
            win_size         = win_size,
            mlp_ratio        = self.mlp_ratio,
            qkv_bias         = qkv_bias,
            qk_scale         = qk_scale,
            drop             = drop_rate,
            attn_drop        = attn_drop_rate,
            drop_path        = dec_dpr[sum(depths[5: 6]): sum(depths[5: 7])],
            norm_layer       = norm_layer,
            use_checkpoint   = use_checkpoint,
            token_projection = token_projection,
            token_mlp        = token_mlp,
            shift_flag       = shift_flag,
            modulator        = modulator,
            cross_modulator  = cross_modulator
        )
        self.upsample_2     = upsample(embed_dim * 8, embed_dim * 2)
        self.decoderlayer_2 = BasicUformerLayer(
            dim              = embed_dim * 4,
            output_dim       = embed_dim * 4,
            input_resolution = (img_size // 2, img_size // 2),
            depth            = depths[7],
            num_heads        = num_heads[7],
            win_size         = win_size,
            mlp_ratio        = self.mlp_ratio,
            qkv_bias         = qkv_bias,
            qk_scale         = qk_scale,
            drop             = drop_rate,
            attn_drop        = attn_drop_rate,
            drop_path        = dec_dpr[sum(depths[5: 7]): sum(depths[5: 8])],
            norm_layer       = norm_layer,
            use_checkpoint   = use_checkpoint,
            token_projection = token_projection,
            token_mlp        = token_mlp,
            shift_flag       = shift_flag,
            modulator        = modulator,
            cross_modulator  = cross_modulator
        )
        self.upsample_3     = upsample(embed_dim * 4, embed_dim)
        self.decoderlayer_3 = BasicUformerLayer(
            dim              = embed_dim * 2,
            output_dim       = embed_dim * 2,
            input_resolution = (img_size, img_size),
            depth            = depths[8],
            num_heads        = num_heads[8],
            win_size         = win_size,
            mlp_ratio        = self.mlp_ratio,
            qkv_bias         = qkv_bias,
            qk_scale         = qk_scale,
            drop             = drop_rate,
            attn_drop        = attn_drop_rate,
            drop_path        = dec_dpr[sum(depths[5: 8]) : sum(depths[5: 9])],
            norm_layer       = norm_layer,
            use_checkpoint   = use_checkpoint,
            token_projection = token_projection,
            token_mlp        = token_mlp,
            shift_flag       = shift_flag,
            modulator        = modulator,
            cross_modulator  = cross_modulator
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"absolute_pos_embed"}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"relative_position_bias_table"}

    def extra_repr(self) -> str:
        return f"embed_dim={self.embed_dim}, token_projection={self.token_projection}, token_mlp={self.mlp},win_size={self.win_size}"

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # Input Projection
        y = self.input_proj(x)
        y = self.pos_drop(y)
        # Encoder
        conv0 = self.encoderlayer_0(y, mask=mask)
        pool0 = self.dowsample_0(conv0)
        conv1 = self.encoderlayer_1(pool0, mask=mask)
        pool1 = self.dowsample_1(conv1)
        conv2 = self.encoderlayer_2(pool1, mask=mask)
        pool2 = self.dowsample_2(conv2)
        conv3 = self.encoderlayer_3(pool2, mask=mask)
        pool3 = self.dowsample_3(conv3)

        # Bottleneck
        conv4 = self.conv(pool3, mask=mask)

        # Decoder
        up0     = self.upsample_0(conv4)
        deconv0 = torch.cat([up0, conv3], -1)
        deconv0 = self.decoderlayer_0(deconv0, mask=mask)

        up1     = self.upsample_1(deconv0)
        deconv1 = torch.cat([up1, conv2], -1)
        deconv1 = self.decoderlayer_1(deconv1, mask=mask)

        up2     = self.upsample_2(deconv1)
        deconv2 = torch.cat([up2, conv1], -1)
        deconv2 = self.decoderlayer_2(deconv2, mask=mask)

        up3     = self.upsample_3(deconv2)
        deconv3 = torch.cat([up3, conv0], -1)
        deconv3 = self.decoderlayer_3(deconv3, mask=mask)

        # Output Projection
        y = self.output_proj(deconv3)
        return x + y if self.dd_in == 3 else y

    def flops(self):
        flops  = 0
        # Input Projection
        flops += self.input_proj.flops(self.reso, self.reso)
        # Encoder
        flops += self.encoderlayer_0.flops() + self.dowsample_0.flops(self.reso, self.reso)
        flops += self.encoderlayer_1.flops() + self.dowsample_1.flops(self.reso // 2, self.reso // 2)
        flops += self.encoderlayer_2.flops() + self.dowsample_2.flops(self.reso // 2 ** 2, self.reso // 2 ** 2)
        flops += self.encoderlayer_3.flops() + self.dowsample_3.flops(self.reso // 2 ** 3, self.reso // 2 ** 3)

        # Bottleneck
        flops += self.conv.flops()

        # Decoder
        flops += self.upsample_0.flops(self.reso // 2 ** 4, self.reso // 2 ** 4) + self.decoderlayer_0.flops()
        flops += self.upsample_1.flops(self.reso // 2 ** 3, self.reso // 2 ** 3) + self.decoderlayer_1.flops()
        flops += self.upsample_2.flops(self.reso // 2 ** 2, self.reso // 2 ** 2) + self.decoderlayer_2.flops()
        flops += self.upsample_3.flops(self.reso // 2,      self.reso // 2)      + self.decoderlayer_3.flops()

        # Output Projection
        flops += self.output_proj.flops(self.reso, self.reso)
        return flops


if __name__ == "__main__":
    input_size = 256
    arch       = Uformer
    depths     = [2, 2, 2, 2, 2, 2, 2, 2, 2]
    model_restoration = Uformer(
        img_size         = input_size,
        embed_dim        = 16,
        depths           = depths,
        win_size         = 8,
        mlp_ratio        = 4.0,
        token_projection = "linear",
        token_mlp        = "leff",
        modulator        = True,
        shift_flag       = False
    )
    print(model_restoration)
    print('# model_restoration parameters: %.2f' % (sum(param.numel() for param in model_restoration.parameters())))
    print("number of GFLOPs: %.2f" % (model_restoration.flops()))
