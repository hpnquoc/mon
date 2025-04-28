#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
References:
    `<https://github.com/Fediory/HVI-CIDNet>`__
"""

import argparse
import copy

import numpy as np
import torch
import torch.optim
import torchvision
from PIL import Image

import mon
from net.cidnet import CIDNet

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
    imgsz        = mon.image_size(imgsz)
    resize       = args.resize
    benchmark    = args.benchmark
    save_image   = args.save_image
    save_debug   = args.save_debug
    keep_subdirs = args.keep_subdirs
    
    # Model
    torch.set_grad_enabled(False)
    model = CIDNet().to(device)
    model.load_state_dict(torch.load(weights, map_location=lambda storage, loc: storage))
    model.eval()
    
    if data == "lol_v1":
        model.trans.gated  = True
    elif data in ["lol_v2_real", "lol_v2_synthetic"]:
        model.trans.gated2 = True
        model.trans.alpha  = 0.8
    else:
        model.trans.alpha  = 0.8
        
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
        mon.console.log(f"FLOPs : {flops:.4f}")
        mon.console.log(f"Params: {params:.4f}")
        mon.console.log(f"Time   = {avg_time:.17f}")
    
    # Data I/O
    mon.console.log(f"[bold red]{data}")
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
                meta       = datapoint["meta"]
                image_path = mon.Path(meta["path"])
                image      = Image.open(image_path).convert("RGB")
                image      = (np.asarray(image) / 255.0)
                image      = torch.from_numpy(image).float()
                image      = image.permute(2, 0, 1)
                image      = image.to(device).unsqueeze(0)
                h0, w0     = mon.image_size(image)
                if resize:
                    image = mon.resize(image, imgsz)
                else:
                    image = mon.resize(image, divisible_by=32)
                
                # Infer
                timer.tick()
                enhanced_image = model(image)
                timer.tock()
                
                # Post-processing
                enhanced_image = torch.clamp(enhanced_image, 0, 1)
                enhanced_image = mon.resize(enhanced_image, (h0, w0))
                
                # Save
                if save_image:
                    output_dir  = mon.parse_output_dir(save_dir, data_name, image_path, keep_subdirs)
                    output_dir.mkdir(parents=True, exist_ok=True)
                    output_path = output_dir / f"{image_path.stem}{mon.SAVE_IMAGE_EXT}"
                    torchvision.utils.save_image(enhanced_image, str(output_path))
        
        avg_time = float(timer.avg_time)
        mon.console.log(f"Average time: {avg_time}")




# ----- Main -----

def main() -> str:
    args = mon.parse_predict_args(model_root=current_dir)
    predict(args)


if __name__ == "__main__":
    main()
