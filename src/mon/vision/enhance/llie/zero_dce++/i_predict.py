#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements the paper: "Learning to Enhance Low-Light Image via Zero-Reference Deep
Curve Estimation," IEEE TPAMI 2022.

References:
    - https://github.com/Li-Chongyi/Zero-DCE_extension
"""

from typing import Sequence

import torch
import torch.optim
import torchvision

import model
import mon

current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# ----- Predict -----
def predict(args: dict) -> str:
    # Parse args
    hostname     = args["hostname"]
    root         = args["root"]
    data         = args["data"]
    fullname     = args["fullname"]
    save_dir     = args["save_dir"]
    weights      = args["weights"]
    device       = args["device"]
    seed         = args["seed"]
    imgsz        = args["imgsz"]
    imgsz        = imgsz[0] if isinstance(imgsz, Sequence) else imgsz
    resize       = args["resize"]
    epochs       = args["epochs"]
    steps        = args["steps"]
    benchmark    = args["benchmark"]
    save_image   = args["save_image"]
    save_debug   = args["save_debug"]
    use_fullpath = args["use_fullpath"]
    verbose      = args["verbose"]
    
    scale_factor = args["network"]["scale_factor"]
    
    # Start
    mon.console.rule(f"[bold red] {fullname}")
    mon.console.log(f"Machine: {hostname}")
    
    # Device
    device = mon.set_device(device)
    
    # Seed
    mon.set_random_seed(seed)
    
    # Data I/O
    mon.console.log(f"[bold red]{data}")
    data_name, data_loader = mon.parse_data_loader(data, root, True, verbose=False)
    
    # Model
    dce_net = model.enhance_net_nopool(scale_factor).to(device)
    dce_net.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    dce_net.eval()
    
    # Benchmark
    if benchmark:
        h = (512 // scale_factor) * scale_factor
        w = (512 // scale_factor) * scale_factor
        flops, params = mon.compute_efficiency_score( model=dce_net, image_size=[h, w])
        mon.console.log(f"FLOPs : {flops:.4f}")
        mon.console.log(f"Params: {params:.4f}")
    
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
                meta       = datapoint["meta"]
                image_path = mon.Path(meta["path"])
                image      = datapoint["image"]
                h0, w0     = mon.image_size(image)
                h1         = (h0 // scale_factor) * scale_factor
                w1         = (w0 // scale_factor) * scale_factor
                image      = image[:, :, 0:h1, 0:w1]
                image      = image.to(device)
                
                # Infer
                timer.tick()
                enhanced, _ = dce_net(image)
                timer.tock()
                
                # Predict
                enhanced = mon.resize(enhanced, (h0, w0))
                
                # Save
                if save_image:
                    if use_fullpath:
                        rel_path    = image_path.relative_path(data_name)
                        output_path = save_dir / rel_path.parent / f"{image_path.stem}.jpg"
                    else:
                        output_path = save_dir / data_name / f"{image_path.stem}.jpg"
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    torchvision.utils.save_image(enhanced, str(output_path))
    
    # Finish
    mon.console.log(f"Average time: {timer.avg_time}")


# ----- Main -----
def main() -> str:
    args = mon.parse_predict_args(model_root=current_dir)
    predict(args)


if __name__ == "__main__":
    main()
