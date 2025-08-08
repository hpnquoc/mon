#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements the paper: "Learning to Enhance Low-Light Image via Zero-Reference Deep
Curve Estimation," IEEE TPAMI 2022.

References:
    - https://github.com/Li-Chongyi/Zero-DCE_extension
"""

import os
import sys

import box
import torch
import torch.optim

import mon

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from model import *

current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# ----- Utils -----
def benchmark(model: torch.nn.Module, imgsz: tuple[int, int]):
    flops, params = mon.compute_efficiency_score(model=model, imgsz=imgsz)
    mon.console.log(f"Params    : {params:.4f}")
    mon.console.log(f"FLOPs     : {flops:.4f}")


# ----- Predict -----
@torch.no_grad()
def predict(args: dict | box.Box) -> str:
    # Start
    mon.print_run_summary(args)

    # Device
    device = mon.set_device(args.device)

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
    scale_factor = args["network"]["scale_factor"]
    model = ZeroDCEpp(scale_factor=scale_factor)
    model.load_state_dict(torch.load(pretrained, weights_only=True))
    model = model.to(device)
    model.eval()
    
    # Benchmark
    if benchmark:
        h = int((512 // scale_factor) * scale_factor)
        w = int((512 // scale_factor) * scale_factor)
        benchmark(model, imgsz=(h, w))
    
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
            path   = mon.Path(datapoint["meta"]["path"])
            image  = datapoint["image"]
            h0, w0 = mon.image_size(image)
            h1     = int((h0 // scale_factor) * scale_factor)
            w1     = int((w0 // scale_factor) * scale_factor)
            image  = image[:, :, 0:h1, 0:w1]
            image  = image.to(device)
            timers.preprocess.tock()

            # Infer
            timers.infer.tick()
            outputs = model(image)
            timers.infer.tock()

            # Postprocess
            timers.postprocess.tick()
            enhanced, _ = outputs
            enhanced    = mon.resize(enhanced, (h0, w0))
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
