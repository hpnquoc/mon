#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""SGZ model prediction pipeline for low-light image enhancement.

References:
    - Paper: "Semantic-Guided Zero-Shot Learning for Low-Light Image/Video
      Enhancement," WACV 2022.
    - Code: https://github.com/ShenZheng2000/Semantic-Guided-Low-Light-Image-Enhancement
"""

import os
import sys

import box
import torch

import mon

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import sgz.utils as utils
from model import *

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

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
    scale_factor = args.network.scale_factor
    model = SGZ(scale_factor, conv_type="dsc")
    model.load_state_dict(torch.load(pretrained, weights_only=True))
    model = model.to(device)
    model.eval()
    
    # Benchmark
    if benchmark:
        h = (512 // scale_factor) * scale_factor
        w = (512 // scale_factor) * scale_factor
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
            path    = mon.Path(datapoint["meta"]["path"])
            image   = utils.image_from_path(str(path))
            h0, w0  = mon.image_size(image)
            # Scale image to have the resolution of multiple of 4
            image   = utils.scale_image(image, scale_factor, device) if scale_factor != 1 else image
            image   = image.to(device)
            timers.preprocess.tock()

            # Infer
            timers.infer.tick()
            outputs = model(image)
            timers.infer.tock()

            # Postprocess
            timers.postprocess.tick()
            enhanced, _ = outputs
            enhanced    = mon.resize(enhanced, (h0, w0), side=None)
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
