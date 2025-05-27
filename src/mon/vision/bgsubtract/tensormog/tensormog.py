#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements the paper: "TensorMoG: A Tensor-Driven Gaussian Mixture Model with
Dynamic Scene Adaptation for Background Modeling," Sensors 2020.
"""

__all__ = [

]

from typing import Literal

import cupy as cp
import numpy as np
import torch

from mon import core, nn
from mon.constants import MLType, MODELS, Task
from mon.vision import geometry, types
from mon.vision.bgsubtract import base

current_file = core.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# ----- Module -----
class MOG:
    """The Mixture of Gaussian Model"""

    def __init__(self, height: int, width: int, num_gaussians: int = 3):
        self.height = height
        self.width  = width

        # Initialize constant parameters
        self.num_gaussians    = num_gaussians
        self.learning_rate    = 0.02
        self.matching_thres   = 2 * 2
        self.background_thres = 0.6
        self.min_var          = 4 * 4 * 3
        self.max_var          = 8 * 8 * 3

        # Initialize the variables
        self.mix_mu     = cp.zeros((self.height, self.width, 3, self.num_gaussians))
        self.mix_var    = cp.zeros((self.height, self.width, 1, self.num_gaussians))
        self.mix_weight = cp.zeros((self.height, self.width, 1, self.num_gaussians))
        self.mix_key    = cp.zeros((self.height, self.width, 1, self.num_gaussians))

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
        u_mask   = cp.logical_and(taken, self.one_hot(cp.argmax(mask * self.mix_key, axis=-1), self.num_gaussians))
        # r_mask is the replacement mask for non-matching mixtures
        r_mask   = cp.logical_and(cp.logical_not(taken), self.one_hot(cp.argmin(self.mix_key, axis=-1), self.num_gaussians))
        nr_mask  = cp.logical_not(r_mask)

        """FINISHING TOUCHES"""
        # weight updates
        w_tmp   = self.mix_weight * (1.0 - self.learning_rate) + u_mask * self.learning_rate
        w_tmp   = nr_mask * w_tmp + r_mask * self.learning_rate
        w_tmp   = w_tmp / cp.sum(w_tmp, axis=-1, keepdims=True)   # normalize the weights
        # mean updates
        mu_tmp  = self.mix_mu + u_mask * d * self.learning_rate
        mu_tmp  = nr_mask * mu_tmp + r_mask * modded_frame
        # var updates
        dv      = u_mask * (d_sum - self.mix_var * 3)
        var_tmp = self.mix_var + dv * self.learning_rate
        # var_tmp = nr_mask * var_tmp + r_mask * self.MINVAR
        var_tmp = cp.maximum(var_tmp, self.min_var)
        var_tmp = cp.minimum(var_tmp, self.max_var)
        # Finally, update the parameters
        self.mix_mu, self.mix_var, self.mix_weight, self.mix_key = [mu_tmp, var_tmp, w_tmp, w_tmp / cp.sqrt(var_tmp)]

    def get_background(self, mode: Literal["numpy", "cupy"] = "numpy"):
        """Get a background based on existing parameters."""
        wmax_ind = cp.argmax(self.mix_key, axis=-1)            # getting indices from distributions with max key
        wmax_hot = self.one_hot(wmax_ind, self.num_gaussians)  # the number of columns
        bg       = cp.max(wmax_hot * self.mix_mu, axis=-1)
        if mode == "cupy":
            return bg.astype(cp.uint8)
        else:
            return cp.asnumpy(bg.astype(cp.uint8))

    def get_foreground(self, image: np.ndarray, background: np.ndarray) -> np.ndarray:
        """Get a mask of difference as a foreground."""
        d 	 = np.abs(image.astype(np.int32) - background.astype(np.int32))
        mask = np.any(d > 80, axis=-1) > 0
        fg   = mask * 255
        return fg.astype(np.uint8)

    def estimate_entropy(self):
        """Returns the model's uncertainty."""
        tmp     = self.mix_weight + (cp.abs(self.mix_weight) < 1e-12)  # we asssume log(0) = 0 to avoid crashing
        entropy = -cp.sum(tmp * cp.log(tmp)) / self.height / self.width
        return entropy

    def estimate_entropy_mask(self):
        # [height, width, 1]
        weights = cp.max(self.mix_weight, axis = -1)
        tmp 	= weights + (cp.abs(weights) < 1e-9)
        entropy = -tmp * cp.log(tmp)
        return entropy

    def estimate_noise_ratio(self):
        return cp.sum(self.estimate_entropy_mask()) / self.height / self.width

    def one_hot(self, x: np.ndarray, depth: int) -> np.ndarray:
        """One-hot encode an array of indices."""
        return cp.eye(depth)[x]


class HVR:
    """The high-variation removal background subtraction model."""

    class State(core.Enum):
        UPDATE   = "update"
        SUSPENSE = "suspense"
        TRAIN    = "train"

    def __init__(
        self,
        height       : int,
        width        : int,
        num_gaussians: int = 3,
        num_updates  : int = 30,
    ):
        self.height        = height
        self.width         = width
        self.num_gaussians = num_gaussians
        self.num_updates   = num_updates

        # Basic variables
        self.state   = self.State.UPDATE
        self.counter = 0

        self.tau               = 0     # average entropy
        self.tau_rate          = 0.01  # entropy rate
        self.tau_updating_rate = 0.025

        # The background
        self.background       = np.zeros((self.height, self.width , 3))
        self.background_model = MOG(self.height, self.width, self.num_gaussians)

    def update(self, image: np.ndarray):
        # Update the background model with a new image.
        if self.state == self.State.UPDATE:
            self.background_model.update(image)

            # Update the stability line.
            entropy = self.estimate_entropy()
            if entropy < self.tau:
                # Defines that the stability line should always be lower or equal
                # to all estimations.
                self.tau = entropy
            else:
                self.tau = self.tau * (1 - self.tau_updating_rate) + entropy * self.tau_updating_rate

            # Recognize an anomaly, of >5% confidence rate
            if entropy - self.tau >= self.tau_rate * self.tau:
                if self.counter == self.num_updates:
                    self.state      = self.State.SUSPENSE  # Move to the SUSPENDING state
                    self.counter    = 0					   # Reset the counter for SUSPENDING
                    self.background = self.background_model.get_background()
                    self.background_model.learning_rate /= 2
                else:
                    self.counter += 1
            else:
                self.counter = 0

        # Update the background model in the SUSPENDING state.
        if self.state == self.State.SUSPENSE:
            self.background_model.update(image)

            # Update the stability line.
            entropy = self.estimate_entropy()
            if entropy < self.tau:
                self.tau = entropy
            else:
                self.tau = self.tau * (1 - self.tau_updating_rate) + entropy * self.tau_updating_rate

            # Recognize stability, within 5% confidence rate
            if entropy - self.tau < self.tau_rate * self.tau:
                if self.counter == self.num_updates:
                    self.state   = self.State.UPDATE
                    self.counter = 0
                    self.background_model.learning_rate *= 2
                else:
                    self.counter += 1
            else:
                self.counter = 0

    def get_background(self) -> np.ndarray:
        """Get the background image."""
        if self.state == self.State.UPDATE:
            self.background = self.background_model.get_background()
        return self.background.astype(np.uint8)

    def get_foreground(self, image: np.ndarray) -> np.ndarray:
        """Get a mask of difference as a foreground."""
        return self.background_model.get_foreground(image, self.get_background())

    def estimate_entropy(self):
        """Returns the model's uncertainty."""
        return self.background_model.estimate_entropy()


# ----- Model -----
@MODELS.register(name="tensormog", arch="tensormog")
class TensorMOG(base.BackgroundSubtractionModel):
    """ZS-N2N model for image denoising.
    
    Args:
        in_channels: The first layer's input channel. Default is ``3`` for RGB image.
        num_channels: Output channels for subsequent layers. Default is ``48``.
    
    References:
        - https://colab.research.google.com/drive/1i82nyizTdszyHkaHBuKPbWnTzao8HF9b?usp=sharing#scrollTo=Srf0GQTYrkxA
    """

    arch     : str          = "tensormog"
    name     : str          = "tensormog"
    tasks    : list[Task]   = [Task.BGSUBTRACT, Task.VIDEO]
    mltypes  : list[MLType] = [MLType.INFERENCE]
    model_dir: core.Path    = current_dir
    zoo      : dict         = {}
    
    def __init__(
        self,
        height       : int = 512,
        width        : int = 512,
        num_gaussians: int = 3,
        num_updates  : int = 30,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.height = height
        self.width  = width
        self.hvr    = HVR(self.height, self.width, num_gaussians, num_updates)

    # ----- Initialize -----
    def init_weights(self, m: nn.Module):
        """Initializes the model's weights.
    
        Args:
            m: ``nn.Module`` to initialize weights for.
        """
        pass
    
    # ----- Forward Pass -----
    def forward(self, datapoint: dict, *args, **kwargs) -> dict:
        """Performs forward pass of the model.
    
        Args:
            datapoint: ``dict`` with datapoint attributes.
    
        Returns:
            ``dict`` of predictions with ``"enhanced"`` keys.
        """
        x = datapoint["image"]
        self.hvr.update(x)
        background = self.hvr.get_background()
        foreground = self.hvr.get_foreground(x)
        return {
            "foreground": foreground,
            "background": background,
        }
    
    # ----- Predict -----
    def infer(self, datapoint: dict, *args, **kwargs) -> dict:
        """Infers model output with optional processing.
    
        Args:
            datapoint: ``dict`` with datapoint attributes.

        Returns:
            ``dict`` of model predictions.
    
        Notes:
            Override for custom pre/post-processing; defaults to ``self.forward()``.
        """
        # Input
        image = datapoint["image"]
        if isinstance(image, torch.Tensor):
            image = types.image_to_array(image, True)

        h0, w0 = types.image_size(image)
        if h0 != self.height or w0 != self.width:
            image = geometry.resize(image, (self.height, self.width))

        # Forward pass
        timer = core.Timer()
        timer.tick()
        outputs = self.forward(datapoint={"image": image})
        timer.tock()
        
        # Post-processing
        foreground = outputs["foreground"]
        background = outputs["background"]
        if h0 != self.height or w0 != self.width:
            foreground = geometry.resize(foreground, (h0, w0))
            background = geometry.resize(background, (h0, w0))

        # Return
        return outputs | {
            "foreground": foreground,
            "background": background,
            "time"      : timer.avg_time,
        }
