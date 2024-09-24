#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import copy

import numpy as np
import torch
import torch.optim
import torchvision
from PIL import Image

import model
import mon

console      = mon.console
current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# region Predict

def predict(args: argparse.Namespace):
    # General config
    data         = args.data
    save_dir     = args.save_dir
    weights      = args.weights
    device       = mon.set_device(args.device)
    imgsz        = args.imgsz
    imgsz        = imgsz[0] if isinstance(imgsz, (list, tuple)) else imgsz
    resize       = args.resize
    benchmark    = args.benchmark
    save_image   = args.save_image
    save_debug   = args.save_debug
    use_fullpath = args.use_fullpath
    
    # Model
    scale_factor = args.scale_factor
    DCE_net      = model.enhance_net_nopool(scale_factor).to(device)
    DCE_net.load_state_dict(torch.load(weights, weights_only=True))
    DCE_net.eval()
    
    # Benchmark
    if benchmark:
        h = (imgsz // scale_factor) * scale_factor
        w = (imgsz // scale_factor) * scale_factor
        flops, params, avg_time = mon.compute_efficiency_score(
            model      = copy.deepcopy(DCE_net),
            image_size = [h, w],
            channels   = 3,
            runs       = 1000,
            use_cuda   = True,
            verbose    = False,
        )
        console.log(f"FLOPs  = {flops:.4f}")
        console.log(f"Params = {params:.4f}")
        console.log(f"Time   = {avg_time:.17f}")
    
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
    with torch.no_grad():
        with mon.get_progress_bar() as pbar:
            for i, datapoint in pbar.track(
                sequence    = enumerate(data_loader),
                total       = len(data_loader),
                description = f"[bright_yellow] Predicting"
            ):
                # Input
                meta          = datapoint.get("meta")
                image_path    = mon.Path(meta["path"])
                data_lowlight = Image.open(image_path).convert("RGB")
                data_lowlight = (np.asarray(data_lowlight) / 255.0)
                data_lowlight = torch.from_numpy(data_lowlight).float()
                h0, w0        = data_lowlight.shape[0], data_lowlight.shape[1]
                h1            = (h0 // scale_factor) * scale_factor
                w1            = (w0 // scale_factor) * scale_factor
                data_lowlight = data_lowlight[0:h1, 0:w1, :]
                data_lowlight = data_lowlight.permute(2, 0, 1)
                data_lowlight = data_lowlight.to(device).unsqueeze(0)
                
                # Infer
                timer.tick()
                enhanced_image, params_maps = DCE_net(data_lowlight)
                timer.tock()
                
                # Predict
                enhanced_image = mon.resize(enhanced_image, (h0, w0))
                
                # Save
                if save_image:
                    if use_fullpath:
                        rel_path    = image_path.relative_path(data_name)
                        output_path = save_dir / rel_path.parent / image_path.name
                    else:
                        output_path = save_dir / data_name / image_path.name
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    torchvision.utils.save_image(enhanced_image, str(output_path))
                
        avg_time = float(timer.avg_time)
        console.log(f"Average time: {avg_time}")

# endregion


# region Main

def main() -> str:
    args = mon.parse_predict_args(model_root=current_dir)
    predict(args)


if __name__ == "__main__":
    main()

# endregion
