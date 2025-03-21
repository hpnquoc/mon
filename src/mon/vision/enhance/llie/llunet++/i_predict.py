#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
References:
    `<https://github.com/xiwang-online/LLUnetPlusPlus>`__
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.optim
import torchvision
from PIL import Image

import mon
from model import NestedUNet

console      = mon.console
current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# region Predict

def predict(args: argparse.Namespace):
    # General config
    hostname     = args.hostname
    root         = args.root
    data         = args.data
    fullname     = args.fullname
    save_dir     = args.save_dir
    weights      = args.weights
    device       = args.device
    seed         = args.seed
    imgsz        = args.imgsz
    resize       = args.resize
    epochs       = args.epochs
    steps        = args.steps
    benchmark    = args.benchmark
    save_image   = args.save_image
    save_debug   = args.save_debug
    use_fullpath = args.use_fullpath
    verbose      = args.verbose
    
    # Start
    console.rule(f"[bold red] {fullname}")
    console.log(f"Machine: {hostname}")
    
    # Device
    device = mon.set_device(device)
    
    # Seed
    mon.set_random_seed(seed)
    
    # Data I/O
    console.log(f"[bold red]{data}")
    data_name, data_loader, data_writer = mon.parse_io_worker(
        src         = data,
        dst         = save_dir,
        to_tensor   = False,
        denormalize = True,
        verbose     = False,
    )
    
    # Model
    model = NestedUNet().to(device)
    model.load_state_dict(torch.load(weights, weights_only=True, map_location=device))
    model.eval()
    
    # Benchmark
    if benchmark:
        flops, params = mon.compute_efficiency_score(model=model, image_size=imgsz)
        console.log(f"FLOPs : {flops:.4f}")
        console.log(f"Params: {params:.4f}")
        
    # Predicting
    timer = mon.Timer()
    with torch.no_grad():
        with mon.get_progress_bar() as pbar:
            for i, datapoint in pbar.track(
                sequence    = enumerate(data_loader),
                total       = len(data_loader),
                description = f"[bright_yellow] Predicting"
            ):
                # Input
                meta       = datapoint.get("meta")
                image_path = mon.Path(meta["path"])
                image      = Image.open(image_path).convert("RGB")
                image      = (np.asarray(image) / 255.0)
                image      = torch.from_numpy(image).float().to(device)
                image      = image.permute(2, 0, 1)
                image      = image.unsqueeze(0)
                h0, w0     = mon.get_image_size(image)
                if resize:
                    image = mon.resize(image, imgsz)
                else:
                    image = mon.resize(image, divisible_by=32)
                
                # Infer
                timer.tick()
                enhanced = model(image)
                timer.tock()
                
                # Post-processing
                enhanced = mon.resize(enhanced, (h0, w0))
                
                # Save
                if save_image:
                    if use_fullpath:
                        rel_path    = image_path.relative_path(data_name)
                        output_path = save_dir / rel_path.parent / f"{image_path.stem}.jpg"
                    else:
                        output_path = save_dir / data_name / f"{image_path.stem}.jpg"
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    torchvision.utils.save_image(enhanced, str(output_path))
    
    # Finish
    console.log(f"Average time: {timer.avg_time}")

# endregion


# region Main

def main() -> str:
    args = mon.parse_predict_args(model_root=current_dir)
    predict(args)


if __name__ == "__main__":
    main()

# endregion
