#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""URetinex-Net++ model prediction pipeline for low-light image enhancement.

References:
    - Paper: "Interpretable Optimization-Inspired Unfolding Network for Low-light
      Image Enhancement," IEEE TPAMI 2025.
    - Code: https://github.com/AndersonYong/URetinex-Net-PLUS
"""

import os
import sys

import box
import torch

import mon

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from model import *

current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# ----- Utils -----
def benchmark(model: torch.nn.Module):
    flops, params = mon.compute_efficiency_score(model=model)
    mon.console.log(f"Params    : {params:.4f}")
    mon.console.log(f"FLOPs     : {flops:.4f}")


def one2three(x):
    return torch.cat([x, x, x], dim=1).to(x)
 
 
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
    data_name, data_loader = mon.parse_data_loader(args.data, args.root, False, verbose=False)

    # Pretrained
    args.Decom_model_high_path         = mon.ZOO_DIR / args.decom_model_high_weights
    args.Decom_model_low_path          = mon.ZOO_DIR / args.decom_model_low_weights
    args.fusion_model_A_path           = mon.ZOO_DIR / args.fusion_weights
    args.pretrain_unfolding_model_path = mon.ZOO_DIR / args.pretrain_unfolding_weights
    '''
    state_dict = torch.load(str(args["fusion_model_A_path"]), weights_only=False)
    print(state_dict.keys())
    print(state_dict["opts"])
    state_dict["opts"].Decom_model_low_path = args["Decom_model_low_path"]
    torch.save(state_dict, str(args["fusion_model_A_path"]))
    
    state_dict = torch.load(str(args["pretrain_unfolding_model_path"]), weights_only=False)
    print(state_dict.keys())
    print(state_dict["opts"])
    state_dict["opts"].Decom_model_low_path = args["Decom_model_low_path"]
    torch.save(state_dict, str(args["pretrain_unfolding_model_path"]))
    '''

    # Model
    model = URetinexNetPP(args)
    model = model.to(device)
    model.eval()
    
    # Benchmark
    if args.benchmark:
        benchmark(model)
    
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
            timers.preprocess.tock()

            # Infer
            timers.infer.tick()
            outputs = model.run(path)
            timers.infer.tock()

            # Postprocess
            timers.postprocess.tick()
            enhanced, _ = outputs
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
