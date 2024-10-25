#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import mon
from mon import albumentation as A
from mon.config import default

current_file = mon.Path(__file__).absolute()


# region Basic

model_name = "llunet++_re"
data_name  = "lol_v1"
root       = current_file.parents[1] / "run"
data_root  = mon.DATA_DIR / "enhance"
project    = None
variant    = None
fullname   = f"{model_name}_{data_name}"
image_size = [384, 384]
seed	   = 1234
verbose    = True

# endregion


# region Model

model = {
	"name"        : model_name,     # The model's name.
	"fullname"    : fullname,       # A full model name to save the checkpoint or weight.
	"root"        : root,           # The root directory of the model.
	"in_channels" : 3,              # The first layer's input channel.
	"out_channels": None,           # A number of classes, which is also the last layer's output channels.
	"weights"     : None,           # The model's weights.
	"loss"        : None,           # Loss function for training the model.
	"loss_weights": [0.40, 0.05, 0.15, 0.40],
	"metrics"     : {
	    "train": None,
		"val"  : [{"name": "psnr"}, {"name": "ssim"}],
		"test" : [{"name": "psnr"}, {"name": "ssim"}],
    },          # A list metrics for validating and testing model.
	"optimizers"  : [
		{
            "optimizer"          : {
				"name"        : "adam",
				"lr"          : 0.00001,
				"weight_decay": 1e-4,
				"betas"       : [0.9, 0.999],
				"eps"		    : 1e-8,
			},
	        "lr_scheduler"       : {
				"scheduler": {
					"name" : "exponential_lr",
					"gamma": 0.99,
				},
				"interval" : "epoch",       # Unit of the scheduler's step size. One of ['step', 'epoch'].
				"frequency": 1,             # How many epochs/steps should pass between calls to `scheduler.step()`.
				"monitor"  : "train/loss",  # Metric to monitor for schedulers like `ReduceLROnPlateau`.
				"strict"   : True,
				"name"     : None,
			},
			"network_params_only": True,
        }
    ],          # Optimizer(s) for training model.
	"debug"       : False,          # If ``True``, run the model in debug mode (when predicting).
	"verbose"     : verbose,        # Verbosity.
}

# endregion


# region Data

data = {
    "name"      : data_name,
    "root"      : data_root,    # A root directory where the data is stored.
    "transform" : A.Compose(transforms=[
		A.Resize(height=image_size[0], width=image_size[1]),
		A.Flip(),
		A.Rotate(),
    ]),  # Transformations performing on both the input and target.
    "to_tensor" : True,         # If ``True``, convert input and target to :class:`torch.Tensor`.
    "cache_data": False,        # If ``True``, cache data to disk for faster loading next time.
    "batch_size": 4,            # The number of samples in one forward pass.
    "devices"   : 0,            # A list of devices to use. Default: ``0``.
    "shuffle"   : True,         # If ``True``, reshuffle the datapoints at the beginning of every epoch.
    "verbose"   : verbose,      # Verbosity.
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
	"devices"          : [0],
	"log_image_every_n_epochs": 1,
	"logger"           : {
		"tensorboard": default.tensorboard,
	},
	"max_epochs"       : 300,
}

# endregion


# region Predicting

predictor = default.predictor | {
	"default_root_dir": root,   # Default path for saving results.
}

# endregion
