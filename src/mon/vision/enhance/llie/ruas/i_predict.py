#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
References:
    https://github.com/KarelZhang/RUAS
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.utils
from PIL import Image

import mon
from model import Network

console      = mon.console
current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# region Predict

def save_images(tensor, path):
    image_numpy = tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)))
    im = Image.fromarray(np.clip(image_numpy * 255.0, 0, 255.0).astype("uint8"))
    im.save(path, 'png')


def predict(args: argparse.Namespace):
    # Parse args
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
    cudnn.benchmark = True
    cudnn.enabled   = True
    
    # Seed
    mon.set_random_seed(seed)
    
    # Data I/O
    console.log(f"[bold red]{data}")
    data_name, data_loader, data_writer = mon.parse_io_worker(
        src         = data,
        dst         = save_dir,
        to_tensor   = True,
        denormalize = True,
        verbose     = False,
    )
    
    # Model
    model = Network().to(device)
    model.load_state_dict(torch.load(str(weights), map_location=device, weights_only=True))
    for p in model.parameters():
        p.requires_grad = False
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
                image      = datapoint.get("image").to(device)
                
                # Infer
                timer.tick()
                u_list, r_list = model(image)
                timer.tock()
                
                # Save
                if save_image:
                    if use_fullpath:
                        rel_path    = image_path.relative_path(data_name)
                        output_path = save_dir / rel_path.parent / f"{image_path.stem}.jpg"
                    else:
                        output_path = save_dir / data_name / f"{image_path.stem}.jpg"
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    save_images(u_list[-1], str(output_path))
                    # save_images(u_list[-1], str(args.output_dir / "lol" / u_name))
                    # save_images(u_list[-2], str(args.output_dir / "dark" / u_name))
                    """
                    if args.model == "lol":
                        save_images(u_list[-1], u_path)
                    elif args.model == "upe" or args.model == "dark":
                        save_images(u_list[-2], u_path)
                    """
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
