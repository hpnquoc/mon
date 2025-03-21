#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Reference:
    https://github.com/pvnieo/Low-light-Image-Enhancement
"""

from __future__ import annotations

import argparse
from typing import Sequence

import cv2

import mon
from exposure_enhancement import enhance_image_exposure

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
    imgsz        = imgsz[0] if isinstance(imgsz, Sequence) else imgsz
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
    
    # Predicting
    timer = mon.Timer()
    with mon.get_progress_bar() as pbar:
        for i, datapoint in pbar.track(
            sequence    = enumerate(data_loader),
            total       = len(data_loader),
            description = f"[bright_yellow] Predicting"
        ):
            # Input
            meta       = datapoint.get("meta")
            image_path = mon.Path(meta["path"])
            image      = datapoint.get("image")
            h0, w0     = mon.get_image_size(image)
            if resize:
                image = cv2.resize(image, (imgsz, imgsz))
            
            # Infer
            timer.tick()
            enhanced = enhance_image_exposure(
                im      = image,
                gamma   = args.gamma,
                lambda_ = args.lambda_,
                dual    = not args.lime,
                sigma   = args.sigma,
                bc      = args.bc,
                bs      = args.bs,
                be      = args.be,
                eps     = args.eps
            )
            timer.tock()
            
            # Post-processing
            if resize:
                enhanced = cv2.resize(enhanced, (w0, h0))
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)
            
            # Save
            if save_image:
                if use_fullpath:
                    rel_path    = image_path.relative_path(data_name)
                    output_path = save_dir / rel_path.parent / f"{image_path.stem}.jpg"
                else:
                    output_path = save_dir / data_name / f"{image_path.stem}.jpg"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(output_path), enhanced)
    
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
