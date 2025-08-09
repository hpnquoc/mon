#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""ZID model prediction pipeline for image dehazing.

References:
    - Paper: "Zero-Shot Image Dehazing," IEEE TIP 2020.
    - Code: https://github.com/XLearning-SCU/2020-TIP-ZID
"""

import os
import sys

import box
import torch.optim

import mon

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from model import *
from zid.utils.image_io import prepare_hazy_image

current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# ----- Utils -----
def benchmark(model: torch.nn.Module):
    flops, params = mon.compute_efficiency_score(model=model)
    mon.console.log(f"Params    : {params:.4f}")
    mon.console.log(f"FLOPs     : {flops:.4f}")


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
            image  = prepare_hazy_image(str(path))
            h0, w0 = mon.image_size(image)
            if args.resize and (h0 != args.imgsz[0] or w0 != args.imgsz[1]):
                image = mon.resize(image, size=args.imgsz)
            timers.preprocess.tock()

            # Save
            out_dir   = mon.parse_output_dir(args.save_dir, data_name, mon.SAVE_IMAGE_DIR, path, args.keep_subdirs, args.save_nearby)
            debug_dir = mon.parse_output_dir(args.save_dir, data_name, mon.SAVE_DEBUG_DIR, path, args.keep_subdirs, args.save_nearby)
            out_dir.mkdir(parents=True, exist_ok=True)
            debug_dir.mkdir(parents=True, exist_ok=True)
            (debug_dir /    "t").mkdir(parents=True, exist_ok=True)
            (debug_dir /    "a").mkdir(parents=True, exist_ok=True)
            (debug_dir / "mask").mkdir(parents=True, exist_ok=True)
            
            # Infer
            timers.infer.tick()
            dh = ZID(str(path.stem), image, args.epochs, clip=True, output_path=str(out_dir))
            dh.optimize()
            dh.finalize()
            timers.infer.tock()
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
