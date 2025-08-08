#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements the paper: "Fast Context-Based Low-Light Image Enhancement via Neural
Implicit Representations," ECCV 2024.

References:
    - https://github.com/ctom2/colie
"""

import os
import sys

import box
import thop
import torch.optim
from fvcore.nn import FlopCountAnalysis, parameter_count

import mon

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from model import *
from utils import *

current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# ----- Utils -----
def compute_efficiency_score(model: torch.nn.Module, imgsz: int = 512) -> tuple[float, float]:
    """Computes FLOPs and parameters for a model.

    Args:
        model: PyTorch model to profile.
        imgsz: Input image size (H, W) or single int. Default is ``512``.

    Returns:
        Tuple of (FLOPs, parameters) as floats.
    """
    patches = torch.rand(imgsz, imgsz, 49).to(mon.get_model_device(model))
    coords  = torch.rand(imgsz, imgsz,  2).to(mon.get_model_device(model))
    
    flops, params = thop.profile(model, inputs=(patches, coords,), verbose=False)
    flops   = FlopCountAnalysis(model, input).total() if flops == 0 else flops
    params  = model.params           if hasattr(model, "params") and params == 0 else params
    params  = parameter_count(model) if hasattr(model, "params") else params
    params  = sum(params.values())   if isinstance(params, dict) else params

    return flops, params


def benchmark(model: torch.nn.Module):
    flops, params = compute_efficiency_score(model=model)
    mon.console.log(f"Params    : {params:.4f}")
    mon.console.log(f"FLOPs     : {flops:.4f}")


# ----- Predict -----
def predict(args: dict | box.Box) -> str:
    window     = args.network.window
    hidden_dim = args.network.hidden_dim
    num_layers = args.network.num_layers
    add_layer  = args.network.add_layer
    L          = args.network.L
    iters      = args.epochs
    
    # Start
    mon.print_run_summary(args)

    # Device
    device = mon.set_device(args.device)

    # Seed
    mon.set_random_seed(args.seed)

    # Data I/O
    data_name, data_loader = mon.parse_data_loader(args.data, args.root, True, verbose=False)
    
    # Model
    model = CoLIE(
        window     = window,
        hidden_dim = hidden_dim,
        num_layers = num_layers,
        add_layer  = add_layer,
        iters      = iters,
        L          = L,
    )
    model = model.to(device)
    
    # Benchmark
    if args.benchmark:
        benchmark(model.model)

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
            path  = mon.Path(datapoint["meta"]["path"])
            image = get_image(str(path)).to(device)
            timers.preprocess.tock()

            # Optimize
            timers.infer.tick()
            outputs = model(image)
            timers.infer.tock()

            # Postprocess
            timers.postprocess.tick()
            enhanced = outputs
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
