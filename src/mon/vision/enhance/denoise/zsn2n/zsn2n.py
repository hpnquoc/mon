#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements the paper: "Zero-Shot Noise2Noise: Efficient Image Denoising without any
Data," CVPR 2023.

References:
    - https://colab.research.google.com/drive/1i82nyizTdszyHkaHBuKPbWnTzao8HF9b?usp=sharing#scrollTo=Srf0GQTYrkxA
"""

__all__ = [
    "ZSN2N",
]

import torch

from mon import core, nn
from mon.constants import MLType, MODELS, Task
from mon.nn import _size_2_t
from mon.vision import geometry, types
from mon.vision.enhance import base

current_file = core.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# ----- Model -----
@MODELS.register(name="zsn2n", arch="zsn2n")
class ZSN2N(base.ImageEnhancementModel):
    """ZS-N2N model for image denoising.
    
    Args:
        in_channels: The first layer's input channel. Default is ``3`` for RGB image.
        num_channels: Output channels for subsequent layers. Default is ``48``.
    
    References:
        - https://colab.research.google.com/drive/1i82nyizTdszyHkaHBuKPbWnTzao8HF9b?usp=sharing#scrollTo=Srf0GQTYrkxA
    """
    
    arch     : str          = "zsn2n"
    name     : str          = "zsn2n"
    tasks    : list[Task]   = [Task.DENOISE]
    mltypes  : list[MLType] = [MLType.ZERO_SHOT]
    model_dir: core.Path    = current_dir
    zoo      : dict         = {}
    
    def __init__(
        self,
        in_channels : int = 3,
        num_channels: int = 48,
        iters       : int = 3000,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.iters = iters
        
        # Network
        self.conv1 = nn.Conv2d(in_channels,  num_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(num_channels, in_channels,  kernel_size=1)
        self.act   = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
        # Optimizer
        self.configure_optimizers()
        
        # Load weights
        if self.weights:
            self.load_weights()
        else:
            self.apply(self.init_weights)
        self.initial_state_dict = self.state_dict()
    
    # ----- Initialize -----
    def init_weights(self, m: nn.Module):
        """Initializes the model's weights.
    
        Args:
            m: ``nn.Module`` to initialize weights for.
        """
        pass
    
    # ----- Forward Pass -----
    def forward_loss(self, datapoint: dict, *args, **kwargs) -> dict:
        """Computes forward pass and loss.
    
        Args:
            datapoint: ``dict`` with datapoint attributes.
    
        Returns:
            ``dict`` of predictions with ``"loss"`` and ``"enhanced"`` keys.
        """
        # Forward
        noisy          = datapoint["image"]
        noisy1, noisy2 = geometry.pair_downsample(noisy)
        datapoint1     = datapoint | {"image": noisy1}
        datapoint2     = datapoint | {"image": noisy2}
        outputs1       = self.forward(datapoint=datapoint1, *args, **kwargs)
        outputs2       = self.forward(datapoint=datapoint2, *args, **kwargs)
        outputs        = self.forward(datapoint=datapoint,  *args, **kwargs)
        
        # Symmetric Loss
        pred1          = noisy1 - outputs1["enhanced"]
        pred2          = noisy2 - outputs2["enhanced"]
        noisy_denoised =  noisy -  outputs["enhanced"]
        denoised1, denoised2 = geometry.pair_downsample(noisy_denoised)
        mse_loss       = nn.MSELoss()
        loss_res       = 0.5 * (mse_loss(noisy1, pred2)     + mse_loss(noisy2, pred1))
        loss_cons      = 0.5 * (mse_loss(pred1,  denoised1) + mse_loss(pred2,  denoised2))
        loss           = loss_res + loss_cons
        
        return outputs | {
			"loss": loss,
		}
    
    def forward(self, datapoint: dict, *args, **kwargs) -> dict:
        """Performs forward pass of the model.
    
        Args:
            datapoint: ``dict`` with datapoint attributes.
    
        Returns:
            ``dict`` of predictions with ``"enhanced"`` keys.
        """
        x = datapoint["image"]
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        y = self.conv3(x)
        if self.predicting:
            y = torch.clamp(y, 0, 1)
        return {"enhanced": y}
    
    # ----- Predict -----
    def infer(
        self,
        datapoint: dict,
        imgsz    : _size_2_t         = 512,
        resize   : bool              = False,
        reset    : bool              = True,
        timers   : core.TimeProfiler = None,
    ) -> dict:
        """Infers model output with optional processing.
    
        Args:
            datapoint: ``dict`` with datapoint attributes.
            imgsz: Input size as ``int`` or [H, W]. Default is ``512``.
            resize: Resize input to ``image_size`` if ``True``. Default is ``False``.
            reset: Whether to reset the weights before training. Default is ``True``.
            timers: ``TimeProfiler`` for measuring time.

        Returns:
            ``dict`` of model predictions.
    
        Notes:
            Override for custom pre/post-processing; defaults to ``self.forward()``.
        """
        # Initialize training components
        if reset:
            self.load_state_dict(self.initial_state_dict, strict=False)
        optimizer    = self.optimizer.get("optimizer",    None)
        lr_scheduler = self.optimizer.get("lr_scheduler", {})
        scheduler    =   lr_scheduler.get("scheduler",    None)
        optimizer    = optimizer or nn.Adam(self, lr=1e-3, weight_decay=0.0001)
        scheduler    = scheduler or nn.StepLR(optimizer, step_size=1000, gamma=0.5)

        # Preprocess
        timers.preprocess.tick() if timers is not None else None
        image  = datapoint["image"]
        h0, w0 = types.image_size(image)
        if resize:
            image = geometry.resize(image, imgsz)
        else:
            image = geometry.resize(image, divisible_by=32)
        image = image.to(self.device)
        timers.preprocess.tock() if timers is not None else None

        # Optimize
        timers.infer.tick() if timers is not None else None
        self.train()
        for _ in range(self.iters):
            outputs = self.forward_loss(datapoint={"image": image})
            optimizer.zero_grad()
            loss = outputs["loss"]
            loss.backward(retain_graph=True)
            optimizer.step()
            scheduler.step()
        self.eval()
        outputs = self.forward(datapoint={"image": image})
        timers.infer.tock() if timers is not None else None
        
        # Postprocess
        timers.postprocess.tick() if timers is not None else None
        enhanced = outputs["enhanced"]
        enhanced = geometry.resize(enhanced, (h0, w0))
        timers.postprocess.tock() if timers is not None else None

        # Return
        return outputs | {
            "enhanced": enhanced,       # Update ``enhanced``
        }
