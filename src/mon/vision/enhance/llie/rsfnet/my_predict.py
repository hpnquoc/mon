#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import copy
import warnings

import numpy as np
import torch
import torch.optim
import torchvision

import mon
from libs.full.src.v8.model import RRNet

eps = np.finfo(np.float32).eps
warnings.filterwarnings("ignore", category=FutureWarning)

console      = mon.console
current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# ----- Predict -----

def predict(args: argparse.Namespace):
    # General config
    data         = args.data
    save_dir     = args.save_dir
    weights      = args.weights
    device       = mon.set_device(args.device)
    imgsz        = args.imgsz
    resize       = args.resize
    benchmark    = args.benchmark
    save_image   = args.save_image
    save_debug   = args.save_debug
    keep_subdirs = args.keep_subdirs
    
    # Model
    args.mode   = "predict"
    args.device = device
    model = RRNet(args)
    model.load_state_dict(torch.load(weights, weights_only=True))
    model.to(device)
    model.eval()
    
    # Benchmark
    if benchmark:
        flops, params, avg_time = mon.compute_efficiency_score(
            model      = copy.deepcopy(model),
            image_size = imgsz,
            channels   = 3,
            runs       = 1000,
            use_cuda   = True,
            verbose    = False,
        )
        console.log(f"FLOPs : {flops:.4f}")
        console.log(f"Params: {params:.4f}")
        console.log(f"Time   = {avg_time:.17f}")
    
    # Data I/O
    console.log(f"[bold red]{data}")
    data_name, data_loader = mon.parse_data_loader(data, root, True, verbose=False)
    
    # Predicting
    timer = mon.Timer()
    with torch.no_grad():
        with mon.create_progress_bar() as pbar:
            for i, datapoint in pbar.track(
                sequence    = enumerate(data_loader),
                total       = len(data_loader),
                description = f"[bright_yellow] Predicting"
            ):
                # Input
                image      = datapoint["image"].to(device).type(torch.float32)
                meta       = datapoint["meta"]
                image_path = mon.Path(meta["path"])
                
                # Infer
                timer.tick()
                enhanced, _ = model(image)
                if args.f_OverExp:
                    enhanced = 1 - enhanced
                timer.tock()
                
                # Save
                if save_image:
                    output_dir  = mon.parse_output_dir(save_dir, data_name, image_path, keep_subdirs)
                    output_dir.mkdir(parents=True, exist_ok=True)
                    output_path = output_dir / f"{image_path.stem}{mon.SAVE_IMAGE_EXT}"
                    torchvision.utils.save_image(enhanced, str(output_path))
        
        avg_time = float(timer.avg_time)
        console.log(f"Average time: {avg_time}")




# ----- Main -----

def main() -> str:
    args = mon.parse_predict_args(model_root=current_dir)
    predict(args)


if __name__ == "__main__":
    main()
