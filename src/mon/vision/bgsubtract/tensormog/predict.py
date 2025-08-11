#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""TensorMoG model prediction pipeline for background subtraction.

References:
    - Paper: "TensorMoG: A Tensor-Driven Gaussian Mixture Model with Dynamic Scene
      Adaptation for Background Modeling," Sensors 2020.
"""

import box
import torch.optim

import mon
from mon.vision.bgsubtract import tensormog

current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# ----- Utils -----
def benchmark(model: torch.nn.Module):
    flops, params = mon.compute_efficiency_score(model=model)
    mon.console.log(f"Params    : {params:.4f}")
    mon.console.log(f"FLOPs     : {flops:.4f}")
    
    
# ----- Predict -----
def predict(args: dict | box.Box) -> str:
    height            = args.network.height
    width             = args.network.width
    num_gaussians     = args.network.num_gaussians
    learning_rate     = args.network.learning_rate
    matching_thres    = args.network.matching_thres
    background_thres  = args.network.background_thres
    num_updates       = args.network.num_updates
    tau_rate          = args.network.tau_rate
    tau_updating_rate = args.network.tau_updating_rate
    
    # Start
    mon.print_run_summary(args)

    # Device
    device = mon.set_device(args.device)
    
    # Seed
    mon.set_random_seed(args.seed)
    
    # Data I/O
    data_name, data_loader = mon.parse_data_loader(args.data, args.root, True, verbose=False)
    
    # Model
    model = tensormog.TensorMOG(
        height            = height,
        width             = width,
        num_gaussians     = num_gaussians,
        learning_rate     = learning_rate,
        matching_thres    = matching_thres,
        background_thres  = background_thres,
        num_updates       = num_updates,
        tau_rate          = tau_rate,
        tau_updating_rate = tau_updating_rate,
    )
    model = model.to(device)
    
    # Benchmark
    if args.benchmark:
        benchmark(model.model)

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
            image  = datapoint["image"]
            h0, w0 = mon.image_size(image)
            if h0 != height or w0 != width:
                image = mon.resize(image, (height, width))
            image = image.to(device)
            timers.preprocess.tock()

            # Optimize
            timers.infer.tick()
            outputs = model(image)
            timers.infer.tock()

            # Postprocess
            timers.postprocess.tick()
            foreground = outputs[0]
            background = outputs[1]
            if h0 != height or w0 != width:
                foreground = mon.resize(foreground, (h0, w0))
                background = mon.resize(background, (h0, w0))
            timers.postprocess.tock()
            
            # Save
            if args.save_image:
                out_dir  = mon.parse_output_dir(args.save_dir, data_name, mon.SAVE_IMAGE_DIR, path, args.keep_subdirs, args.save_nearby)
                out_path = out_dir / f"{path.stem}{mon.SAVE_IMAGE_EXT}"
                mon.save_image(background, out_path)
            
            # Save Debug
            if args.save_debug:
                debug_dir  = mon.parse_output_dir(args.save_dir, data_name, mon.SAVE_DEBUG_DIR, path, args.keep_subdirs, args.save_nearby)
                debug_path = debug_dir / f"{path.stem}_foreground{mon.SAVE_IMAGE_EXT}"
                mon.save_image(foreground, debug_path)
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
