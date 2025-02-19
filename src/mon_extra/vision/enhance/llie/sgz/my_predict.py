#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import copy
import os

import torch
import torchvision

import mon
import utils
from modeling import model as mmodel

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # For GPU only

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
    imgsz        = imgsz[0] if isinstance(imgsz, list | tuple) else imgsz
    resize       = args.resize
    benchmark    = args.benchmark
    save_image   = args.save_image
    save_debug   = args.save_debug
    use_fullpath = args.use_fullpath
    scale_factor = args.scale_factor
    
    # Model
    net          = mmodel.enhance_net_nopool(scale_factor, conv_type="dsc").to(device)
    net.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    net.eval()
    
    # Benchmark
    if benchmark:
        h = (imgsz // scale_factor) * scale_factor
        w = (imgsz // scale_factor) * scale_factor
        # h, w = imgsz, imgsz
        flops, params, avg_time = mon.compute_efficiency_score(
            model      = copy.deepcopy(net),
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
                data_lowlight = utils.image_from_path(str(image_path))
                h0, w0        = mon.get_image_size(meta["shape"])
                # Scale image to have the resolution of multiple of 4
                data_lowlight = utils.scale_image(data_lowlight, scale_factor, device) if scale_factor != 1 else data_lowlight
                data_lowlight = data_lowlight.to(device)
                
                # Infer
                timer.tick()
                enhanced_image, params_maps = net(data_lowlight)
                timer.tock()
                
                # Post-processing
                enhanced_image = mon.resize(enhanced_image, (h0, w0), side=None)
                
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
    args.weights = args.weights
    predict(args)


if __name__ == "__main__":
    main()

# endregion
