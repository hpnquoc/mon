#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""NeurOP.

This module implements the paper: "Neural Color Operators for Sequential Image
Retouching".

References:
    - https://github.com/amberwangyili/neurop
"""

from __future__ import annotations

from typing import Sequence

import imageio
import torch

import mon
from models import build_model
from utils import *

console      = mon.console
current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# region Predict

def predict(args: dict) -> str:
    # Parse args
    hostname     = args["hostname"]
    root         = args["root"]
    data         = args["data"]
    fullname     = args["fullname"]
    save_dir     = args["save_dir"]
    weights      = args["weights"]
    device       = args["device"]
    seed         = args["seed"]
    imgsz        = args["imgsz"]
    imgsz        = imgsz[0] if isinstance(imgsz, Sequence) else imgsz
    resize       = args["resize"]
    epochs       = args["epochs"]
    steps        = args["steps"]
    benchmark    = args["benchmark"]
    save_image   = args["save_image"]
    save_debug   = args["save_debug"]
    use_fullpath = args["use_fullpath"]
    verbose      = args["verbose"]
    
    opt_path       = str(current_dir / "options" / "test" / args["opt_path"])
    opt            = parse(opt_path)
    opt            = dict_to_nonedict(opt)
    opt["dist"]    = False
    opt["device"]  = device
    opt["weights"] = weights
    
    # Start
    console.rule(f"[bold red] {fullname}")
    console.log(f"Machine: {hostname}")
    
    # Device
    device = mon.set_device(device)
    
    # Seed
    mon.set_random_seed(seed)
    
    # Data I/O
    console.log(f"[bold red]{data}")
    data_name, data_loader = mon.parse_data_loader(data, root, True, verbose=False)
    
    # Model
    model = build_model(opt)
    
    # Benchmark
    if benchmark:
        flops, params = model.measure_efficiency_score(image_size=imgsz)
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
                meta       = datapoint["meta"]
                image_path = mon.Path(meta["path"])
                image      = datapoint["image"].to(device)
                h0, w0     = mon.get_image_size(image)
                if resize:
                    image = mon.resize(image, imgsz)
                else:
                    image = mon.resize(image, divisible_by=32)
                
                # Infer
                timer.tick()
                model.feed_data(data = {
                    "LQ": image,
                    "GT": image,
                })
                model.test()
                timer.tock()
                
                # Post-processing
                visuals = model.get_current_visuals()
                sr_img  = visuals["rlt"]
                h1, w1  = mon.get_image_size(sr_img)
                if h1 != h0 or w1 != w0:
                    sr_img = mon.resize(sr_img, (h0, w0))
                    
                # Save
                if save_image:
                    if use_fullpath:
                        rel_path    = image_path.relative_path(data_name)
                        output_path = save_dir / rel_path.parent / f"{image_path.stem}.jpg"
                    else:
                        output_path = save_dir / data_name / f"{image_path.stem}.jpg"
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    imageio.imwrite(str(output_path), (255.0 * sr_img).astype("uint8"))
        
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
