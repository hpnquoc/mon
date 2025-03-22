#!/usr/bin/edenoised1nv python
# -*- coding: utf-8 -*-

from __future__ import annotations

from mon.config import default

import mon

current_file = mon.Path(__file__).absolute()


# region Basic

model_name = "zero_linr"
data_name  = ""
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
	# Model
	"in_channels"      : 3,              # The first layer's input channel.
	"out_channels"     : None,           # A number of classes, which is also the last layer's output channels.
	"mapping_func"     : "pvde",         # One of: ["p", "v", "d", "e", "pv", "pd", "pe", "pvde"]. Default: "pvde".
	"window_size"      : 1,              # Context window size. Default: 1.
	"down_size"        : 256,            # Input image size. Default: 256.
	"num_layers"       : 4,              # Total layer's depth. Default: 4.
	"add_layers"       : 2,              # Number of layers for output branch. Default: 2.
	"omega_0"          : 30.0,           # Default: 30.0.
	"first_bias_scale" : 20.0,           # For "finer". Default: 20.0.
	"s_nonlinear"      : "finer",        # Activation function for the spatial branch. Default: "finer".
	"use_ff"           : True,           # Use Fourier Feature embedding for the spatial branch. Default: True.
	"ff_gaussian_scale": 10.0,           # For Fourier Feature embedding. Default: 10.0.
	"v_nonlinear"      : "finer",        # Activation function for the pixel value branch. Default: "sine".
	"reduce_channels"  : False,          # Reduce the output channels of input encoders.
	"depth_threshold"  : 0.7,            # For adjusting the learned residual. Default: 0.7.
	"edge_threshold"   : 0.05,           # Edge threshold. Default: 0.05.
	# Post-process
	"gf_radius"        : 1,              # Radius of the guided filter. Default: 1.
	"use_denoise"      : False,          # Use denoising. Default: False.
	"denoise_ksize"    : (3, 3),         # For denoising. Default: (3, 3).
    "denoise_color"    : 0.1,            # For denoising. Default: 0.1.
    "denoise_space"    : (1.5, 1.5),     # For denoising. Default: (1.5, 1.5).
	# Loss
	"loss_e_mean"      : 0.1,            # Default: 0.1.
	"loss_w_f"         : 1,              # Default: 1.
	"loss_w_s"         : 5,              # Default: 5.
	"loss_w_e"         : 8,              # Default: 8.
	"loss_w_tv"        : 20,             # Default: 20.
	"loss_w_de"        : 1,              # Default: 1.
	"loss_w_c"         : 5,              # Default: 5.
	# Training
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
	"max_epochs"       : 100,
}

# endregion


# region Predicting

predictor = default.predictor | {
	"default_root_dir": root,  # Default path for saving results.
	"save_debug"      : True,  # Save debug images.
}

# endregion
