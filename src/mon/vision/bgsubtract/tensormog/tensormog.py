#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements the paper: "TensorMoG: A Tensor-Driven Gaussian Mixture Model with
Dynamic Scene Adaptation for Background Modeling," Sensors 2020.
"""

__all__ = [

]

import abc
from typing import Literal

import cupy as cp
import numpy as np
import torch

from mon import core, nn
from mon.constants import MLType, MODELS, Task
from mon.nn import _size_2_t
from mon.vision import geometry, types
from mon.vision.enhance import base

current_file = core.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# ----- Utils -----
def one_hot(a: np.ndarray, depth: int) -> np.ndarray:
    """One-hot encode an array of indices.

    Args:
        a: an array of int indices.
        depth: the number of components.
    """
    return cp.eye(depth)[a]


# ----- Background Module -----
class BackgroundModel(abc.ABC):
    """Abstract base class for background models."""

    def __init__(self, height: int, width: int):
        self.height = height
        self.width  = width

    @abc.abstractmethod
    def bg_subtraction(self, image: np.ndarray):
        pass


class MOG(BackgroundModel):
    """The Mixture of Gaussian Model"""

    def __init__(self, height: int, width: int, num_gaussian: int = 3):
        super().__init__(height=height, width=width)
        # Constant parameters
        self.num_gaussian     = num_gaussian
        self.learning_rate    = 0.02
        self.matching_thres   = 2 * 2
        self.background_thres = 0.6
        self.min_var          = 4 * 4 * 3
        self.max_var          = 8 * 8 * 3

        self.mix_mu     = cp.zeros((height, width, 3, self.num_gaussian))
        self.mix_var    = cp.zeros((height, width, 1, self.num_gaussian))
        self.mix_weight = cp.zeros((height, width, 1, self.num_gaussian))
        self.mix_key    = cp.zeros((height, width, 1, self.num_gaussian))

    def deep_copy(self, other_mog: "MOG"):
        """Makes another model this model's deep copy.

        Args:
            other_mog: another MOG object.
        """
        other_mog.mix_mu     = self.mix_mu.copy()
        other_mog.mix_var    = self.mix_var.copy()
        other_mog.mix_weight = self.mix_weight.copy()
        other_mog.mix_key    = self.mix_key.copy()

    def shallow_copy(self, other_mog: "MOG"):
        """Makes another model this model's shallow copy.

        Args:
            other_mog: another MOG object.
        """
        other_mog.mix_mu     = self.mix_mu
        other_mog.mix_var    = self.mix_var
        other_mog.mix_weight = self.mix_weight
        other_mog.mix_key    = self.mix_key

    def reset_model_complete(self):
        """Reset the model completely."""
        # Reset GMM to all zeros
        self.mix_mu     = cp.zeros(( self.height, self.width, 3, self.num_gaussian))
        self.mix_var    = cp.zeros(( self.height, self.width, 1, self.num_gaussian))
        self.mix_weight = cp.zeros(( self.height, self.width, 1, self.num_gaussian))
        self.mix_key    = cp.zeros(( self.height, self.width, 1, self.num_gaussian))

    def reset_model_with_components(self, nod: int):
        """Remove (self.num_gaussian - nod) least important variables.

        Args:
            nod: number of distributions to keep.

        Returns:
            None. The model is resetted partially.
            It retains at most nod distributions for each pixel.
        """
        assert 0 < nod <= self.num_gaussian
        nord = self.num_gaussian - nod  # number of removed distributions
        tmp  = cp.expand_dims(cp.partition(self.mix_key, nord-1, axis=-1)[:, :, :, nord-1], -1)
        mask = self.mix_key > tmp  # mask is binary, 1 to keep, 0 to remove

        self.mix_mu     = self.mix_mu     * mask
        self.mix_var    = self.mix_var    * mask
        self.mix_weight = self.mix_weight * mask
        self.mix_weight = self.mix_weight / cp.sum(self.mix_weight, axis=-1, keepdims=True)  # renormalize
        self.mix_key    = self.mix_weight / (self.mix_var + 1e-9)

    def update_init(self, image: np.ndarray):
        """Trains the initial background model."""
        self.update(image)

    def update(self, image: np.ndarray):
        """Updates the background model."""
        modded_frame = cp.expand_dims(cp.array(image), -1)

        """CREATION OF MASKS"""
        d        = modded_frame - self.mix_mu
        d_square = d * d
        d_sum    = cp.sum(d_square, axis=2, keepdims=True)

        mask     = d_sum < (self.matching_thres * self.mix_var)
        taken    = cp.any(mask, axis=-1, keepdims=True)

        # u_mask is the update mask for matching mixtures
        u_mask   = cp.logical_and(taken, one_hot(cp.argmax(mask * self.mix_key, axis=-1), self.num_gaussian))

        # r_mask is the replacement mask for non-matching mixtures
        r_mask   = cp.logical_and(cp.logical_not(taken), one_hot(cp.argmin(self.mix_key, axis=-1), self.num_gaussian))
        nr_mask  = cp.logical_not(r_mask)

        """FINISHING TOUCHES"""
        # weight updates
        w_tmp    = self.mix_weight * (1.0 - self.learning_rate) + u_mask * self.learning_rate
        w_tmp    = nr_mask * w_tmp + r_mask * self.learning_rate
        w_tmp    = w_tmp / cp.sum(w_tmp, axis=-1, keepdims=True)   # normalize the weights
        # mean updates
        mu_tmp   = self.mix_mu + u_mask * d * self.learning_rate
        mu_tmp   = nr_mask * mu_tmp + r_mask * modded_frame
        # var updates
        dv       = u_mask * (d_sum - self.mix_var * 3)
        var_tmp  = self.mix_var + dv * self.learning_rate
        # var_tmp = nr_mask * var_tmp + r_mask * self.MINVAR
        var_tmp  = cp.maximum(var_tmp, self.min_var)  # upper cap
        var_tmp  = cp.minimum(var_tmp, self.max_var)  # upper cap
        # Finally, update the parameters
        self.mix_mu, self.mix_var, self.mix_weight, self.mix_key = [mu_tmp, var_tmp, w_tmp, w_tmp / cp.sqrt(var_tmp)]

    def bg_subtraction(self, ascp: bool = False, mode: Literal["cupy", "numpy"] = "cupy"):
        """Performs background subtraction."""
        wmax_ind = cp.argmax(self.mix_key, axis=-1)      # getting indices from distributions with max key
        wmax_hot = one_hot(wmax_ind, self.num_gaussian)  # the number of columns
        # the background is then
        bg = cp.max(wmax_hot * self.mix_mu, axis=-1)

        if mode == "cupy":
            return cp.asnumpy(bg.astype(cp.uint8))
        else:
            return bg.astype(np.uint8)

    def calculate_foreground_img(self, image: np.ndarray, background: np.ndarray) -> np.ndarray:
        """Obtain the foreground image based on the input image and background.

        Args:
            image: Input image.
            background: Background image.
        """
        d          = np.abs(image.astype(np.int32) - background.astype(np.int32))
        mask       = np.any(d > 80, axis=-1) > 0
        foreground = mask * 255
        return foreground.astype(np.uint8)

    def close(self):
        pass

    def entropy_estimate(self):
        """Returns the model's uncertainty."""
        tmp     = self.mix_weight + (cp.abs(self.mix_weight) < 1e-12)  # we asssume log(0) = 0 to avoid crashing
        entropy = -cp.sum(tmp * cp.log(tmp)) / self.height / self.width
        return entropy

    def entropy_mask_estimate(self):
        # [height, width, 1]
        weights = cp.max(self.mix_weight, axis = -1)
        tmp     = weights + (cp.abs(weights) < 1e-9)
        entropy = -tmp * cp.log(tmp)
        return entropy

    def estimate_noise_ratio(self):
        return cp.sum(self.entropy_mask_estimate()) / self.height / self.width

    def backup_model(self):
        """Returns shallow copies of model parameters."""
        return [
            cp.copy(self.mix_mu),
            cp.copy(self.mix_var),
            cp.copy(self.mix_weight),
            cp.copy(self.mix_key)
        ]

    def restore_model(self, backup_model):
        """Assign model parameters to that of a backup model."""
        self.mix_mu, self.mix_var, self.mix_weight, self.mix_key = backup_model


# ----- HVR Module -----


# ----- Model -----
@MODELS.register(name="tensormog", arch="tensormog")
class TensorMOG(base.ImageEnhancementModel):
    """ZS-N2N model for image denoising.
    
    Args:
        in_channels: The first layer's input channel. Default is ``3`` for RGB image.
        num_channels: Output channels for subsequent layers. Default is ``48``.
    
    References:
        - https://colab.research.google.com/drive/1i82nyizTdszyHkaHBuKPbWnTzao8HF9b?usp=sharing#scrollTo=Srf0GQTYrkxA
    """
    
    arch     : str          = "tensormog"
    name     : str          = "tensormog"
    tasks    : list[Task]   = [Task.DENOISE, Task.VIDEO]
    mltypes  : list[MLType] = [MLType.INFERENCE]
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
        noisy1, noisy2 = self.pair_downsampler(noisy)
        datapoint1     = datapoint | {"image": noisy1}
        datapoint2     = datapoint | {"image": noisy2}
        outputs1       = self.forward(datapoint=datapoint1, *args, **kwargs)
        outputs2       = self.forward(datapoint=datapoint2, *args, **kwargs)
        outputs        = self.forward(datapoint=datapoint,  *args, **kwargs)
        
        # Symmetric Loss
        pred1          = noisy1 - outputs1["enhanced"]
        pred2          = noisy2 - outputs2["enhanced"]
        noisy_denoised =  noisy -  outputs["enhanced"]
        denoised1, denoised2 = self.pair_downsampler(noisy_denoised)
        mse_loss  = nn.MSELoss()
        loss_res  = 0.5 * (mse_loss(noisy1, pred2)    + mse_loss(noisy2, pred1))
        loss_cons = 0.5 * (mse_loss(pred1, denoised1) + mse_loss(pred2, denoised2))
        loss      = loss_res + loss_cons
        
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
        datapoint    : dict,
        image_size   : _size_2_t = 512,
        resize       : bool      = False,
        reset_weights: bool      = True,
    ) -> dict:
        """Infers model output with optional processing.
    
        Args:
            datapoint: ``dict`` with datapoint attributes.
            image_size: Input size as ``int`` or [H, W]. Default is ``512``.
            resize: Resize input to ``image_size`` if ``True``. Default is ``False``.
            reset_weights: Whether to reset the weights before training. Default is ``True``.
            
        Returns:
            ``dict`` of model predictions.
    
        Notes:
            Override for custom pre/post-processing; defaults to ``self.forward()``.
        """
        # Initialize training components
        if reset_weights:
            self.load_state_dict(self.initial_state_dict, strict=False)
        optimizer    = self.optimizer.get("optimizer",    None)
        lr_scheduler = self.optimizer.get("lr_scheduler", {})
        scheduler    =   lr_scheduler.get("scheduler",    None)
        optimizer = optimizer or nn.Adam(self, lr=1e-3, weight_decay=0.0001)
        scheduler = scheduler or nn.StepLR(optimizer, step_size=1000, gamma=0.5)
        
        # Input
        image  = datapoint["image"].to(self.device)
        h0, w0 = types.image_size(image)
        if resize:
            image = geometry.resize(image, image_size)
        else:
            image = geometry.resize(image, divisible_by=32)
        
        # Optimize
        timer = core.Timer()
        timer.tick()
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
        timer.tock()
        
        # Post-processing
        enhanced = outputs["enhanced"]
        enhanced = geometry.resize(enhanced, (h0, w0))
        
        # Return
        return outputs | {
            "enhanced": enhanced,
            "time"    : timer.avg_time,
        }
