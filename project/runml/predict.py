#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements prediction pipeline."""

from __future__ import annotations

import socket

import torch

import mon

console      = mon.console
current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# region Predict

def predict(args: dict) -> str:
    # Get arguments
    fullname     = args["fullname"]
    imgsz        = mon.parse_hw(args["image_size"])
    seed         = args["seed"]
    save_dir     = args["predictor"]["default_root_dir"]
    source       = args["predictor"]["source"]
    devices      = args["predictor"]["devices"] or "auto"
    resize       = args["predictor"]["resize"]
    benchmark    = args["predictor"]["benchmark"]
    save_image   = args["predictor"]["save_image"]
    save_debug   = args["predictor"]["save_debug"]
    use_data_dir = args["predictor"]["use_data_dir"]
    use_fullpath = args["predictor"]["use_fullpath"]
    console.rule(f"[bold red] {fullname}")
    
    # Seed
    mon.set_random_seed(seed)
    
    # Device
    devices = torch.device(("cpu" if not torch.cuda.is_available() else devices))
    
    # Model
    model: mon.Model = mon.MODELS.build(config=args["model"])
    model = model.to(devices)
    model.eval()
    
    # Benchmark
    if benchmark and torch.cuda.is_available() and hasattr(model, "compute_efficiency_score"):
        flops, params, avg_time = model.compute_efficiency_score(
            image_size = imgsz,
            channels   = 3,
            runs       = 100,
            verbose    = False,
        )
        console.log(f"FLOPs  = {flops:.4f}")
        console.log(f"Params = {params:.4f}")
        console.log(f"Time   = {avg_time:.4f}")
    
    # Data I/O
    console.log(f"[bold red] {source}")
    data_name, data_loader, data_writer = mon.parse_io_worker(
        src         = source,
        dst         = save_dir,
        to_tensor   = True,
        denormalize = True,
        verbose     = False,
    )
    save_dir = save_dir if save_dir not in [None, "None", "", "."] else model.root
    
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
                datapoint = datapoint,
                imgsz     = imgsz,
                resize    = resize,
            )
            time = outputs.pop("time", None)
            if time:
                run_time.append(time)
            
            # Save
            if save_image:
                _,   output = outputs.popitem()
                if use_fullpath:
                    rel_path   = image_path.relative_path(data_name)
                    output_dir = save_dir / rel_path.parent
                else:
                    output_dir = save_dir / data_name
                output_path  = output_dir / f"{meta['stem']}.png"
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
                            path = debug_output_dir / f"{meta['stem']}_{k}.png"
                            mon.write_image(path, v)
    
    # Finish
    avg_time = float(sum(run_time) / len(run_time))
    console.log(f"Average time: {avg_time}")
    return str(save_dir)
        
# endregion


# region Main

def parse_predict_args(model_root: str | mon.Path = None) -> dict:
    hostname = socket.gethostname().lower()
    
    # Get input args
    input_args = vars(mon.parse_predict_input_args())
    config     = input_args.get("config")
    root       = mon.Path(input_args.get("root"))

    # Get config args
    config = mon.parse_config_file(
        project_root = root,
        model_root   = model_root,
        weights_path = None,
        config       = config,
    )
    args   = mon.load_config(config)
    
    # Prioritize input args --> config file args
    arch         = input_args.get("arch")
    model        = input_args.get("model")        or args.get("model_name")
    data         = input_args.get("data")         or args["predictor"]["source"]
    project      = input_args.get("project")      or args.get("project")
    variant      = input_args.get("variant")      or args.get("variant")
    fullname     = input_args.get("fullname")     or args.get("fullname")
    save_dir     = input_args.get("save_dir")     # or args["predictor"]["default_root_dir"]
    weights      = input_args.get("weights")      or args["model"]["weights"]
    devices      = input_args.get("device")       or args["predictor"]["devices"]
    imgsz        = input_args.get("imgsz")        or args.get("image_size")
    resize       = input_args.get("resize")       or args.get("resize")
    benchmark    = input_args.get("benchmark")    or args.get("benchmark")
    save_image   = input_args.get("save_image")   or args.get("save_image")
    save_debug   = input_args.get("save_debug")   or args.get("save_debug")
    use_data_dir = input_args.get("use_data_dir") or args.get("use_data_dir")
    use_fullpath = input_args.get("use_fullpath") or args.get("use_fullpath")
    verbose      = input_args.get("verbose")      or args.get("verbose")
    extra_args   = input_args.get("extra_args")
    
    # Parse arguments
    save_dir = save_dir or mon.parse_save_dir(root/"run"/"predict", arch, model, None, project, variant)
    save_dir = mon.Path(save_dir)
    weights  = mon.to_list(weights)
    weights  = None       if isinstance(weights, list | tuple) and len(weights) == 0 else weights
    weights  = weights[0] if isinstance(weights, list | tuple) and len(weights) == 1 else weights
    
    # Update arguments
    args["hostname"]    = hostname
    args["config"]      = config
    args["arch"]        = arch
    args["root"]        = root
    args["project"]     = project
    args["variant"]     = variant
    args["fullname"]    = fullname
    args["image_size"]  = imgsz
    args["verbose"]     = verbose
    args["model"]      |= {
        "root"      : save_dir,
        "fullname"  : fullname,
        "weights"   : weights,
        "loss"      : None,
        "metrics"   : None,
        "optimizers": None,
        "verbose"   : verbose,
    }
    args["predictor"]  |= {
        "default_root_dir": save_dir,
        "source"          : data,
        "devices"         : devices,
        "resize"          : resize,
        "benchmark"       : benchmark,
        "save_image"      : save_image,
        "save_debug"      : save_debug,
        "use_data_dir"    : use_data_dir,
        "use_fullpath"    : use_fullpath,
        "verbose"         : verbose,
    }
    
    save_dir.mkdir(parents=True, exist_ok=True)
    if config and config.is_config_file():
        mon.copy_file(src=config, dst=save_dir / f"config{config.suffix}")
        
    return args


def main():
    args = parse_predict_args()
    predict(args)


if __name__ == "__main__":
    main()

# endregion
