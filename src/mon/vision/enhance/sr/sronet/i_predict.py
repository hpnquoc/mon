#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse

import torch
import torch.optim
import torchvision

import models
import mon
from utils import make_coord

console      = mon.console
current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# region Predict

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
    scale        = args.scale
    scale_max    = args.scale_max
    
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
        to_tensor   = True,
        denormalize = True,
        verbose     = False,
    )
    
    # Model
    model = models.make(torch.load(weights, weights_only=True)["model"], load_sd=True).to(device)
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
                meta        = datapoint.get("meta")
                image_path  = mon.Path(meta["path"])
                image       = datapoint.get("image").to(device)
                h           = int(image.shape[-2] * int(scale))
                w           = int(image.shape[-1] * int(scale))
                scale_      = h / image.shape[-2]
                coord       = make_coord((h, w), flatten=False).to(device)
                cell        = torch.ones(1, 2).to(device)
                cell[:, 0] *= 2 / h
                cell[:, 1] *= 2 / w
                cell_factor = max(scale_ / scale_max, 1)
                
                # Infer
                timer.tick()
                pred = model(
                    inp   = ((image - 0.5) / 0.5).to(device),
                    coord = coord.unsqueeze(0),
                    cell  = cell_factor * cell
                )#.squeeze(0)
                pred = (pred * 0.5 + 0.5).clamp(0, 1).reshape(1, 3, h, w).cpu()
                timer.tock()
                
                # Save
                if save_image:
                    if use_fullpath:
                        rel_path    = image_path.relative_path(data_name)
                        output_path = save_dir / rel_path.parent / f"{image_path.stem}.jpg"
                    else:
                        output_path = save_dir / data_name / f"{image_path.stem}.jpg"
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    torchvision.utils.save_image(pred, str(output_path))
    
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
