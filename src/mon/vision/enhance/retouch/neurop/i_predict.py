#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements the paper: "Neural Color Operators for Sequential Image Retouching,"
ECCV 2022.

References:
    - https://github.com/amberwangyili/neurop
"""

import box
import imageio
import torch

import mon
from models import build_model
from utils import *

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
    cfg_path        = current_dir / "option" / "test" / args.cfg
    cfgs            = parse(str(cfg_path))
    cfgs            = dict_to_nonedict(cfgs)
    cfgs["dist"]    = False
    cfgs["weights"] = args.weights

    # Start
    mon.print_run_summary(args)

    # Device
    device = mon.set_device(args.device)
    cfgs["device"] = device

    # Seed
    mon.set_random_seed(args.seed)

    # Data I/O
    data_name, data_loader = mon.parse_data_loader(args.data, args.root, True, verbose=False)

    # Model
    model = build_model(cfgs)
    
    # Benchmark
    if args.benchmark:
        benchmark(model)
    
    # Predicting
    timers = mon.TimeProfiler()
    with mon.create_progress_bar() as pbar:
        for i, datapoint in pbar.track(
            sequence    = enumerate(data_loader),
            total       = len(data_loader),
            description = f"[bright_yellow]Predicting"
        ):
            # Preprocess
            timers.preprocess.tick()
            path   = mon.Path(datapoint["meta"]["path"])
            image  = datapoint["image"]
            h0, w0 = mon.image_size(image)
            if args.resize and h0 != args.imgsz[0] and w0 != args.imgsz[1]:
                image = mon.resize(image, size=args.imgsz)
            else:
                image = mon.resize(image, divisible_by=32)
            image  = image.to(device)
            timers.preprocess.tock()

            # Infer
            timers.infer.tick()
            model.feed_data(data = {
                "LQ": image,
                "GT": image,
            })
            model.test()
            timers.infer.tock()
            
            # Postprocess
            timers.postprocess.tick()
            outputs  = model.get_current_visuals()
            enhanced = outputs["rlt"]
            h1, w1   = mon.image_size(enhanced)
            if h1 != h0 or w1 != w0:
                enhanced = mon.resize(enhanced, (h0, w0))
            enhanced = (255.0 * enhanced).astype("uint8")
            timers.postprocess.tock()

            # Save
            if args.save_image:
                out_dir  = mon.parse_output_dir(args.save_dir, data_name, mon.SAVE_IMAGE_DIR, path, args.keep_subdirs, args.save_nearby)
                out_path = out_dir / f"{path.stem}{mon.SAVE_IMAGE_EXT}"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                imageio.imwrite(str(out_path), enhanced)
        
    # Finish
    timers.print()
    return str(args.save_dir)
    

# ----- Main -----
def main() -> str:
    args = mon.parse_predict_args(model_root=current_dir)
    predict(args)


if __name__ == "__main__":
    main()
