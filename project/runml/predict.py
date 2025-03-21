#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Predict Pipeline.

This script predicts the output of a model on a given dataset.
"""

from __future__ import annotations

import argparse

import mon

console      = mon.console
current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# region Predict

def predict(args: argparse.Namespace) -> str:
    # Parse args
    args         = vars(args)
    hostname     = args["hostname"]
    root         = args["root"]
    data         = args["data"]
    fullname     = args["fullname"]
    save_dir     = args["save_dir"]
    weights      = args["weights"]
    device       = args["device"] or "auto"
    seed         = args["seed"]
    imgsz        = args["imgsz"]
    resize       = args["resize"]
    # epochs       = args["epochs"]
    # steps        = args["steps"]
    benchmark    = args["benchmark"]
    save_image   = args["save_image"]
    save_debug   = args["save_debug"]
    use_fullpath = args["use_fullpath"]
    verbose      = args["verbose"]
    
    # Start
    console.rule(f"[bold red] {fullname}")
    console.log(f"Machine: {hostname}")
    
    # Device
    device = mon.set_device(device)
    
    # Seed
    mon.set_random_seed(seed)
    
    # Data I/O
    console.log(f"[bold red] {data}")
    data_name, data_loader, data_writer = mon.parse_io_worker(
        src         = data,
        dst         = save_dir,
        to_tensor   = True,
        denormalize = True,
        verbose     = False,
    )
    
    # Model
    args["modelmodule"] |= {
        "fullname"  : fullname,
        "root"      : save_dir,
        "weights"   : weights,
        "optimizers": None,  # Skip initialization for efficiency
        "loss"      : None,  # Skip initialization for efficiency
        "metrics"   : None,  # Skip initialization for efficiency
        "debug"     : save_debug,
        "verbose"   : verbose,
    }
    model: mon.Model = mon.MODELS.build(config=args["modelmodule"])
    model = model.to(device)
    model.eval()
    
    # Benchmark
    if benchmark and hasattr(model, "compute_efficiency_score"):
        flops, params = model.compute_efficiency_score(image_size=imgsz)
        console.log(f"FLOPs  = {flops:.4f}")
        console.log(f"Params = {params:.4f}")
        
    # Predicting
    run_time = []
    with mon.get_progress_bar() as pbar:
        for i, datapoint in pbar.track(
            sequence    = enumerate(data_loader),
            total       = len(data_loader),
            description = f"[bright_yellow] Predicting"
        ):
            # Input
            meta       = datapoint.get("meta")
            image_path = mon.Path(meta["path"])
            
            # Infer
            outputs = model.infer(
                datapoint  = datapoint,
                image_size = imgsz,
                resize     = resize,
            )
            time = outputs.pop("time", None)
            if time:
                run_time.append(time)
            
            # Save image
            if save_image:
                _, output = outputs.popitem()
                if use_fullpath:
                    rel_path   = image_path.relative_path(data_name)
                    output_dir = save_dir / rel_path.parent
                else:
                    output_dir = save_dir / data_name
                output_path  = output_dir / f"{meta['stem']}.jpg"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                mon.write_image(output_path, output)
            # Save video
            if data_writer:
                data_writer.write_batch(frames=output)
            # Save Debug
            if save_debug:
                if use_fullpath:
                    rel_path         = image_path.relative_path(data_name)
                    debug_output_dir = save_dir / rel_path.parents[1] / f"{rel_path.parent.name}_debug"
                else:
                    debug_output_dir = save_dir / f"{data_name}_debug"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                for k, v in outputs.items():
                    if mon.is_image(v):
                        path = debug_output_dir / f"{meta['stem']}_{k}.jpg"
                        mon.write_image(path, v)
    
    # Finish
    avg_time = float(sum(run_time) / len(run_time)) if run_time else 0
    console.log(f"Average time: {avg_time}")
    return str(save_dir)
        
# endregion


# region Main

def main():
    args = mon.parse_predict_args()
    predict(args)


if __name__ == "__main__":
    main()

# endregion
