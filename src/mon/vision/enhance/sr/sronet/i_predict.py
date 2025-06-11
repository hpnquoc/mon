#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements the paper: "Super-Resolution Neural Operator," CVPR 2023.

References:
    - https://github.com/2y7c3/Super-Resolution-Neural-Operator
"""

import box
import torch
import torch.optim

import models
import mon
from utils import make_coord

current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# ----- Utils -----
def benchmark(model: torch.nn.Module):
    flops, params = mon.compute_efficiency_score(model=model)
    mon.console.log(f"FLOPs : {flops:.4f}")
    mon.console.log(f"Params: {params:.4f}")


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
    model = models.make(torch.load(pretrained, weights_only=True)["model"], load_sd=True)
    model = model.to(device)
    model.eval()

    # Benchmark
    if args.benchmark:
        benchmark(model)
        
    # Predict
    timers = mon.TimeProfiler()
    with mon.create_progress_bar() as pbar:
        for i, datapoint in pbar.track(
            sequence    = enumerate(data_loader),
            total       = len(data_loader),
            description = f"[bright_yellow]Predicting"
        ):
            # Preprocess
            timers.preprocess.tick()
            path        = mon.Path(datapoint["meta"]["path"])
            image       = datapoint["image"]
            image       = image.to(device)
            h0          = int(image.shape[-2] * int(args.scale))
            w0          = int(image.shape[-1] * int(args.scale))
            scale_      = h0 / image.shape[-2]
            coord       = make_coord((h0, w0), flatten=False).to(device)
            cell        = torch.ones(1, 2).to(device)
            cell[:, 0] *= 2 / h0
            cell[:, 1] *= 2 / w0
            cell_factor = max(scale_ / args.scale_max, 1)
            timers.preprocess.tock()

            # Infer
            timers.infer.tick()
            outputs = model(
                inp   = ((image - 0.5) / 0.5).to(device),
                coord = coord.unsqueeze(0),
                cell  = cell_factor * cell
            )
            timers.infer.tock()

            # Postprocess
            timers.postprocess.tick()
            enhanced = (outputs * 0.5 + 0.5).clamp(0, 1).reshape(1, 3, h0, w0).cpu()
            timers.postprocess.tock()

            # Save
            if args.save_image:
                out_dir  = mon.parse_output_dir(args.save_dir, data_name, mon.SAVE_IMAGE_DIR, path, args.keep_subdirs, args.save_nearby)
                out_path = out_dir / f"{path.stem}{mon.SAVE_IMAGE_EXT}"
                mon.save_image(enhanced, out_path)
    
    # Finish
    timers.print()
    return str(args.save_dir)


# ----- Main -----
def main() -> str:
    args = mon.parse_predict_args(model_root=current_dir)
    predict(args)


if __name__ == "__main__":
    main()
