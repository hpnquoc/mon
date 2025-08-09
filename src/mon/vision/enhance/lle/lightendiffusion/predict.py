#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""LightenDiffusion model prediction pipeline for low-light image enhancement.

References:
    - Paper: "LightenDiffusion: Unsupervised Low-Light Image Enhancement with
      Latent-Retinex Diffusion Models," ECCV 2024.
    - Code: https://github.com/JianghaiSCU/LightenDiffusion
"""

import argparse
import os
import sys

import box
import numpy as np
import torch
import yaml

import mon
from mon.nn import functional as F

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from model import *

current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# ----- Utils -----
def benchmark(model: torch.nn.Module):
    flops, params = mon.compute_efficiency_score(model=model, channels=6)
    mon.console.log(f"Params    : {params:.4f}")
    mon.console.log(f"FLOPs     : {flops:.4f}")


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


# ----- Predict -----
@torch.no_grad()
def predict(args: dict | box.Box) -> str:
    cfg_path = current_dir / "lightendiffusion" / "option" / args.cfg
    with open(str(cfg_path), "r") as f:
        cfgs = yaml.safe_load(f)
    cfgs = dict2namespace(cfgs)

    # Start
    mon.print_run_summary(args)

    # Device
    device      = mon.set_device(args.device)
    cfgs.device = device

    # Seed
    mon.set_random_seed(args.seed)

    # Data I/O
    data_name, data_loader = mon.parse_data_loader(args.data, args.root, True, verbose=False)
    
    # Pretrained
    pretrained = args.resume
    if args.weights and args.weights.is_weights_file(exist=True):
        pretrained = args.weights
    if pretrained and pretrained.is_weights_file(exist=True):
        mon.console.log(f"Pretrained: {pretrained}.")
    else:
        raise ValueError(f"Invalid weights file: {pretrained}.")

    # Model
    model_args = argparse.Namespace(**{
        "mode"        : "evaluation",
        "resume"      : str(pretrained),
        "image_folder": str(args.save_dir),
    })
    diffusion = LightenDiffusion(model_args, cfgs)
    diffusion.load_ddm_ckpt(str(pretrained), ema=False)
    diffusion.model.eval()

    # Benchmark
    if args.benchmark:
        benchmark(diffusion.model)
    
    # Predict
    timers = mon.TimeProfiler()
    timers.total.tick()
    with mon.create_progress_bar() as pbar:
        for i, datapoint in pbar.track(
            sequence    = enumerate(data_loader),
            total       = len(data_loader),
            description = f"[bright_yellow]Predicting"
        ):
             # Preprocess
            timers.preprocess.tick()
            path     = mon.Path(datapoint["meta"]["path"])
            image    = datapoint["image"]
            h0, w0   = mon.image_size(image)
            img_h_64 = int(64 * np.ceil(h0 / 64.0))
            img_w_64 = int(64 * np.ceil(w0 / 64.0))
            x_cond   = F.pad(image, (0, img_w_64 - w0, 0, img_h_64 - h0), "reflect")
            x_cond   = x_cond.to(device)
            image    = image.to(device)
            timers.preprocess.tock()

            # Infer
            timers.infer.tick()
            outputs = diffusion.model(torch.cat((x_cond, x_cond), dim=1))["pred_x"]
            timers.infer.tock()

            # Postprocess
            timers.postprocess.tick()
            enhanced = outputs[:, :, :h0, :w0]
            timers.postprocess.tock()

            # Save
            if args.save_image:
                out_dir  = mon.parse_output_dir(args.save_dir, data_name, mon.SAVE_IMAGE_DIR, path, args.keep_subdirs, args.save_nearby)
                out_path = out_dir / f"{path.stem}{mon.SAVE_IMAGE_EXT}"
                mon.save_image(enhanced, out_path)
    timers.total.tock()

    # Finish
    timers.print()
    return str(args.save_dir)


# ----- Main -----
def main() -> str:
    args = mon.parse_predict_args(model_root=current_dir)
    predict(args)


if __name__ == "__main__":
    main()
