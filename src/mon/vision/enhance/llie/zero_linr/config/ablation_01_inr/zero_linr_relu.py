#!/usr/bin/edenoised1nv python
# -*- coding: utf-8 -*-

from __future__ import annotations

import mon
from mon.config import default

current_file = mon.Path(__file__).absolute()


# region Basic

model_name = "zero_linr_relu"
data_name  = "fivek_e"
root       = current_file.parents[1] / "run"
data_root  = mon.DATA_DIR / "enhance"
project    = None
variant    = None
fullname   = f"{model_name}_{data_name}"
image_size = [512, 512]
seed	   = 100
verbose    = True

# endregion


# region Model

model = {
	"name"             : model_name,     # The model's name.
	"fullname"         : fullname,       # A full model name to save the checkpoint or weight.
	"root"             : root,           # The root directory of the model.
	"in_channels"      : 3,              # The first layer's input channel.
	"out_channels"     : None,           # A number of classes, which is also the last layer's output channels.
	"color_space"      : "hsv_d",        # Color space. Best: hsv_d
	"window_size"      : [3, 5, 7],      # Context window size.
	"hidden_channels"  : 256,            # Hidden channels.
	"down_size"        : 256,            # Downsampling size.
	"hidden_layers"    : 2,              # Number of hidden layers.
	"out_layers"       : 1,              # Number of output layers.
	"omega_0"          : 30.0,           # Default: 30.0
	"first_bias_scale" : None,           # Default: None
	"nonlinear"        : "relu",         # Non-linear activation.
	"use_ff"           : False,          # Default: False
	"ff_gaussian_scale": None,           # Default: None
	"edge_threshold"   : 0.05,           # Edge threshold. Default: 0.05
	"depth_gamma"	   : 0.5,            # Depth gamma. Default: 0.5
	"gf_radius"        : 3,              # Radius of the guided filter. Default: 3
	"use_denoise"      : False,          # If ``True``, use denoising. Default: False
	"denoise_ksize"    : (3, 3),         # Default: (3, 3)
    "denoise_color"    : 0.1,            # Default: 0.1
    "denoise_space"    : (1.5, 1.5),     # Default: (1.5, 1.5)
	"loss_hsv"         : True,           # If ``True``, use HSV loss. Default: True
	"exp_mean"         : 0.7,            # Default: 0.7
	"exp_weight"       : 8.0,            # Default: 8.0
	"spa_weight"	   : 1.0,            # Default: 1.0
	"tv_weight"        : 20.0,           # Default: 20.0
	"spar_weight"	   : 5.0,            # Default: 5.0
	"depth_weight"     : 1.0,            # Default: 1.0
	"edge_weight"      : 1.0,            # Default: 1.0
	"color_weight"     : 5.0,            # Default: 5.0
	"weights"          : None,           # The model's weights.
	"metrics"          : {
	    "train": None,
		"val"  : [{"name": "psnr"}, {"name": "ssim"}],
		"test" : [{"name": "psnr"}, {"name": "ssim"}],
    },          # A list metrics for validating and testing model.
	"optimizers"       : [
		{
            "optimizer"          : {
	            "name"        : "adam",
	            "lr"          : 0.00005,
	            "weight_decay": 0.00001,
	            "betas"       : [0.9, 0.99],
			},
			"lr_scheduler"       : None,
			"network_params_only": True,
        }
    ],          # Optimizer(s) for training model.
	"debug"            : False,          # If ``True``, run the model in debug mode (when predicting).
	"verbose"          : verbose,        # Verbosity.
}

# endregion


# region Data

data = {
    "name"      : data_name,
    "root"      : data_root,     # A root directory where the data is stored.
	"transform" : None,          # Transformations performing on both the input and target.
    "to_tensor" : True,          # If ``True``, convert input and target to :class:`torch.Tensor`.
    "cache_data": False,         # If ``True``, cache data to disk for faster loading next time.
    "batch_size": 1,             # The number of samples in one forward pass.
    "devices"   : 0,             # A list of devices to use. Default: ``0``.
    "shuffle"   : True,          # If ``True``, reshuffle the datapoints at the beginning of every epoch.
    "verbose"   : verbose,       # Verbosity.
}

# endregion


# region Training

trainer = default.trainer | {
	"callbacks"        : [
		default.log_training_progress,
		default.model_checkpoint | {
			"filename": fullname,
			"monitor" : "val/psnr",
			"mode"    : "max",
		},
		default.model_checkpoint | {
			"filename" : fullname,
			"monitor"  : "val/ssim",
			"mode"     : "max",
			"save_last": True,
		},
		default.learning_rate_monitor,
		default.rich_model_summary,
		default.rich_progress_bar,
	],
	"default_root_dir" : root,  # Default path for logs and weights.
	"log_image_every_n_epochs": 1,
	"logger"           : {
		"tensorboard": default.tensorboard,
	},
	"max_epochs"       : 200,
}

# endregion


# region Predicting

predictor = default.predictor | {
	"default_root_dir": root,  # Default path for saving results.
	"save_debug"      : True,  # Save debug images.
}

# endregion
