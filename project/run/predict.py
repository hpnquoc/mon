#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Predicts model output on a given dataset."""

import box
import torch

import mon

current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# ----- Utils -----
def benchmark(model: mon.Model):
    if hasattr(model, "compute_efficiency_score"):
        flops, params = model.compute_efficiency_score()
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
        mon.console.log(f"Pretrained: {None}, training from scratch.")

    # Model
    args["modelmodule"] |= {
        "fullname" : args.fullname,
        "root"     : args.save_dir,
        "weights"  : pretrained,
        # "optimizer": None,  # Skip initialization for efficiency
        # "loss"     : None,  # Skip initialization for efficiency
        # "metrics"  : None,  # Skip initialization for efficiency
        "debug"    : args.save_debug,
        "verbose"  : args.verbose,
    }
    model: mon.Model = mon.MODELS.build(config=args.modelmodule)
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
            # Input
            path = mon.Path(datapoint["meta"]["path"])
            
            # Infer
            outputs = model.infer(
                datapoint  = datapoint,
                image_size = args.imgsz,
                resize     = args.resize,
                timers     = timers,
            )
            if "timers" in outputs:
                timer = outputs.pop("timers")
                timers.preprocess.append(timer.preprocess.avg_time)
                timers.infer.append(timer.infer.avg_time)
                timers.postprocess.append(timer.postprocess.avg_time)

            # Save
            if args.save_image:
                k, output = outputs.popitem()
                out_dir   = mon.parse_output_dir(args.save_dir, data_name, mon.SAVE_IMAGE_DIR, path, args.keep_subdirs, args.save_nearby)
                out_path  = out_dir / f"{path.stem}{mon.SAVE_IMAGE_EXT}"
                mon.save_image(output, out_path)
            
            # Save Debug
            if args.save_debug:
                debug_dir = mon.parse_output_dir(args.save_dir, data_name, mon.SAVE_DEBUG_DIR, path, args.keep_subdirs, args.save_nearby)
                for k, v in outputs.items():
                    if mon.is_image(v):
                        path = debug_dir / f"{path.stem}_{k}{mon.SAVE_IMAGE_EXT}"
                        mon.save_image(v, path)
    
    # Finish
    timers.print()
    return str(args.save_dir)


# ----- Main -----
def main():
    args = mon.parse_predict_args()
    predict(args)


if __name__ == "__main__":
    main()
