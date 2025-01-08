#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""ESDNet.

This module implements the paper: "Learning A Spiking Neural Network for
Efficient Image Deraining," IJCAI 2024.

References:
    https://github.com/MingTian99/ESDNet
"""

from __future__ import annotations

__all__ = [

]

import math
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from mon import core, nn
from mon.globals import MODELS, Scheme, Task
from mon.vision import geometry
from mon.vision.dtype import image as I
from mon.vision.enhance import base
from spikingjelly.activation_based import functional, layer, neuron

console      = core.console
current_file = core.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# region Module

class FeatureRefinementBlock(nn.Module):
    
    def __init__(self, channel: int, reduction: int):
        super().__init__()
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.sa = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 1, 1),
            nn.Conv2d(channel, channel // 8, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, channel, 3, 1, 1),
            nn.Sigmoid()
        )
	
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.ca(x)
        t = self.sa(x)
        s = torch.mul((1 - t), a) + torch.mul(t, x)
        return s


class OverlapPatchEmbed(nn.Module):
    
    def __init__(self, in_channels: int = 3, embed_dim: int = 32, bias: bool = False):
        super().__init__()
        functional.set_step_mode(self, step_mode="m")
        self.proj = layer.Conv2d(in_channels, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)
	
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x


class SpikingResidualBlock(nn.Module):
    
    def __init__(self, dim: int):
        super().__init__()
        v_th  = 0.15
        alpha = 1 / (2 ** 0.5)
        functional.set_step_mode(self, step_mode="m")
        self.residual = nn.Sequential(
            neuron.LIFNode(v_threshold=v_th, backend="cupy", step_mode="m", decay_input=False),
	        layer.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False, step_mode="m"),
	        layer.ThresholdDependentBatchNorm2d(num_features=dim, alpha=alpha, v_th=v_th, affine=True),
			
            neuron.LIFNode(v_threshold=v_th, backend="cupy", step_mode="m", decay_input=False),
	        layer.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False, step_mode="m"),
	        layer.ThresholdDependentBatchNorm2d(num_features=dim, alpha=alpha, v_th=v_th * 0.2, affine=True),
        )
        self.shortcut = nn.Sequential(
	        layer.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False, step_mode="m"),
	        layer.ThresholdDependentBatchNorm2d(num_features=dim, alpha=alpha, v_th=v_th, affine=True),
        )
        self.attn = layer.MultiDimensionalAttention(T=4, reduction_t=4, reduction_c=16, kernel_size=3, C=dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = torch.clone(x)
        out      = self.residual(x) + self.shortcut(x)
        out      = self.attn(out) + shortcut
        return out
    

class DownSampling(nn.Module):
    
    def __init__(self, dim: int):
        super().__init__()
        v_th  = 0.15
        alpha = 1 / (2 ** 0.5)
        functional.set_step_mode(self, step_mode="m")
        self.maxpool_conv = nn.Sequential(
	        neuron.LIFNode(v_threshold=v_th, backend="cupy", step_mode="m", decay_input=False),
	        layer.Conv2d(dim, dim * 2, kernel_size=3, stride=2, padding=1, step_mode="m", bias=False),
	        layer.ThresholdDependentBatchNorm2d(alpha=alpha, v_th=v_th, num_features=dim * 2, affine=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class UpSampling(nn.Module):
    
    def __init__(self, dim):
        super().__init__()
        v_th  = 0.15
        alpha = 1 / (2 ** 0.5)
        self.scale_factor = 2
        self.up = nn.Sequential(
            neuron.LIFNode(v_threshold=v_th, backend="cupy", step_mode="m", decay_input=False),
	        layer.Conv2d(dim, dim // 2, kernel_size=3, stride=1, padding=1, step_mode="m", bias=False),
	        layer.ThresholdDependentBatchNorm2d(alpha=alpha, v_th=v_th, num_features=dim // 2, affine=True),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        temp   = torch.zeros((input.shape[0], input.shape[1], input.shape[2], input.shape[3] * self.scale_factor, input.shape[4] * self.scale_factor)).cuda()
        output = []
        for i in range(input.shape[0]):
            temp[i] = F.interpolate(input[i], scale_factor=self.scale_factor, mode="bilinear")
            output.append(temp[i])
        out = torch.stack(output, dim=0)
        return self.up(out)
    
# endregion


# region Model

@MODELS.register(name="esdnet", arch="esdnet")
class ESDNet(base.ImageEnhancementModel):
    """Learning A Spiking Neural Network for Efficient Image Deraining.
    
    References:
        https://github.com/MingTian99/ESDNet
    """
    
    model_dir: core.Path    = current_dir
    arch     : str          = "esdnet"
    tasks    : list[Task]   = [Task.DERAIN, Task.LLIE]
    schemes  : list[Scheme] = [Scheme.SUPERVISED]
    zoo      : dict         = {}

    def __init__(
        self,
        in_channels   : int       = 3,
        out_channels  : int       = 3,
        dim           : int       = 24,
        enc_num_blocks: list[int] = [4, 4, 6, 6],
        dec_num_blocks: list[int] = [4, 4, 6, 6],
        bias          : bool      = False,
        T             : int       = 4,
        weights       : Any       = None,
        *args, **kwargs
    ):
        super().__init__(
            name        = "esdnet",
            in_channels = in_channels,
            weights     = weights,
            *args, **kwargs
        )
        self.in_channels   = in_channels
        self.out_channels  = out_channels
        self.dim           = dim
        self.en_num_blocks = enc_num_blocks
        self.de_num_blocks = dec_num_blocks
        
        # Construct model
        v_th  = 0.15
        alpha = 1 / (2 ** 0.5)
        functional.set_backend(self,   backend="cupy")
        functional.set_step_mode(self, step_mode="m")
        
        self.T = T
        self.patch_embed    = OverlapPatchEmbed(in_channels=in_channels, embed_dim=dim)
        self.encoder_level1 = nn.Sequential(*[
            SpikingResidualBlock(dim=int(dim * 1)) for i in range(enc_num_blocks[0])
        ])
        
        self.down1_2        = DownSampling(dim)  # From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[
            SpikingResidualBlock(dim=int(dim * 2 ** 1)) for i in range(enc_num_blocks[1])
        ])
        
        self.down2_3        = DownSampling(int(dim * 2 ** 1))  # From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[
            SpikingResidualBlock(dim=int(dim * 2 ** 2)) for i in range(enc_num_blocks[2])
        ])
        
        self.decoder_level3 = nn.Sequential(*[
            SpikingResidualBlock(dim=int(dim * 2 ** 2)) for i in range(dec_num_blocks[2])
        ])
        
        self.up3_2 = UpSampling(int(dim * 2 ** 2))  # From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Sequential(
            neuron.LIFNode(v_threshold=v_th, backend="cupy", step_mode="m", decay_input=False),
            layer.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias, step_mode="m"),
            layer.ThresholdDependentBatchNorm2d(num_features=int(dim * 2 ** 1), alpha=alpha, v_th=v_th),
        )
        self.decoder_level2 = nn.Sequential(*[
            SpikingResidualBlock(dim=int(dim * 2 ** 1)) for i in range(dec_num_blocks[1])
        ])
        
        self.up2_1          = UpSampling(int(dim * 2 ** 1))  # From Level 2 to Level 1 (NO 1x1 conv to reduce channels)
        self.decoder_level1 = nn.Sequential(*[
            SpikingResidualBlock(dim=int(dim * 2 ** 1)) for i in range(dec_num_blocks[0])
        ])

        self.refinement = FeatureRefinementBlock(channel=int(dim * 2 ** 1), reduction=8)
        self.output     = nn.Sequential(nn.Conv2d(in_channels=int(dim * 2 ** 1), out_channels=out_channels, kernel_size=3, stride=1, padding=1))
        
        # Loss
        self.loss = nn.PSNRLoss(reduction="mean")
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)

    def init_weights(self, m: nn.Module):
        pass
    
    def forward(self, datapoint: dict, *args, **kwargs) -> dict:
        self.assert_datapoint(datapoint)
        input = datapoint.get("image")
        short = input.clone()
        # Repeat Feature
        if len(input.shape) < 5:
            input = (input.unsqueeze(0)).repeat(self.T, 1, 1, 1, 1)
        
        functional.reset_net(self)
        inp_enc_level1 = self.patch_embed(input)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)
        
        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)
        
        out_dec_level3 = self.decoder_level3(out_enc_level3)
        
        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], dim=2)
        
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)
        
        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], dim=2)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        # Image Reconstruction
        out_dec_level1 = self.refinement(out_dec_level1.mean(0))
        enhanced       = (self.output(out_dec_level1)) + short
        
        return {
            "enhanced": enhanced
        }
    
    def infer(
        self,
        datapoint   : dict,
        image_size  : int | Sequence[int] = 512,
        crop_size   : int = 80,
        overlap_size: int = 8,
        resize      : bool = False,
        *args, **kwargs
    ) -> dict:
        """Infer the model on a single datapoint. This method is different from
        :obj:`forward()` in term that you may want to perform additional
        pre-processing or post-processing steps.
        
        Notes:
            If you want to perform specific pre-processing or post-processing
            steps, you should override this method.
        
        Args:
            datapoint: A :obj:`dict` containing the attributes of a datapoint.
            image_size: The input size. Default: ``512``.
            crop_size: The crop size. Default: ``80``.
            overlap_size: The overlap size. Default: ``8``
            resize: Resize the input image to the model's input size.
                Default: ``False``.
        """
        # Pre-processing
        self.assert_datapoint(datapoint)
        image = datapoint.get("image")
        image = image.to(self.device)
        b, c, h0, w0 = image.shape
        '''
        for k, v in datapoint.items():
            if I.is_image(v):
                if resize:
                    datapoint[k] = geometry.resize(v, image_size)
                else:
                    datapoint[k] = geometry.resize(v, divisible_by=32)
        for k, v in datapoint.items():
            if isinstance(v, torch.Tensor):
                datapoint[k] = v.to(self.device)
        image = datapoint.get("image")
        '''
        
        # Forward
        timer = core.Timer()
        timer.tick()
        split_data, starts = self.split_image(image, crop_size=crop_size, overlap_size=overlap_size)
        for j, data in enumerate(split_data):
            data          = data.to(self.device)
            split_data[j] = self.forward(datapoint={"image": data}, *args, **kwargs)["enhanced"]
            split_data[j] = split_data[j].cpu()
            functional.reset_net(self)
        enhanced = self.merge_image(split_data, starts, shape=(b, c, crop_size, crop_size))
        enhanced = torch.clamp(enhanced, 0, 1)
        timer.tock()
        outputs  = {"enhanced": enhanced}
        self.assert_outputs(outputs)
        
        # Post-processing
        '''
        for k, v in outputs.items():
            if I.is_image(v):
                h1, w1 = I.get_image_size(v)
                if h1 != h0 or w1 != w0:
                    outputs[k] = geometry.resize(v, (h0, w0))
        '''
        
        # Return
        outputs["time"] = timer.avg_time
        return outputs
    
    def split_image(
        self,
        image       : torch.Tensor,
        crop_size   : int = 80,
        overlap_size: int = 8
    ) -> tuple[list, list]:
        b, c, h, w = image.shape
        
        h_starts   = [x for x in range(0, h, crop_size - overlap_size)]
        while h_starts[-1] + crop_size >= h:
            h_starts.pop()
        h_starts.append(h - crop_size)
        
        w_starts   = [x for x in range(0, w, crop_size - overlap_size)]
        while w_starts[-1] + crop_size >= w:
            w_starts.pop()
        w_starts.append(w - crop_size)
       
        starts     = []
        split_data = []
        for hs in h_starts:
            for ws in w_starts:
                img_data = image[:, :, hs:hs + crop_size, ws:ws + crop_size]
                starts.append((hs, ws))
                split_data.append(img_data)
        return split_data, starts
    
    def get_score_map(self, b, c, h, w, is_mean: bool = True) -> torch.Tensor:
        center_h = h / 2
        center_w = w / 2
        score    = torch.ones((b, c, h, w))
        if not is_mean:
            for h in range(h):
                for w in range(w):
                    score[:, :, h, w] = 1.0 / (math.sqrt((h - center_h) ** 2 + (w - center_w) ** 2 + 1e-3))
        return score
    
    def merge_image(self, split_data, starts, shape=(1, 3, 80, 80)) -> torch.Tensor:
        b, c, h, w  = shape[0], shape[1], shape[2], shape[3]
        total_score = torch.zeros((b, c, h, w))
        merge_img   = torch.zeros((b, c, h, w))
        score_map   = self.get_score_map(b, c, h, w, is_mean=False)
        for simg, cstart in zip(split_data, starts):
            hs, ws = cstart
            merge_img[:, :, hs:hs + h, ws:ws + w]   += score_map * simg
            total_score[:, :, hs:hs + h, ws:ws + w] += score_map
        merge_img = merge_img / total_score
        return merge_img

# endregion
