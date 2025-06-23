#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements the paper: "DarkIR: Robust Low-Light Image Restoration," CVPR 2025.

References:
    - https://github.com/cidautai/DarkIR
"""

import box
import torch.optim
import torchvision
from ptflops import get_model_complexity_info

import mon
from archs import *
from utils.test_utils import *

current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# ----- Utils -----
def benchmark(model: torch.nn.Module):
    flops, params  = mon.compute_efficiency_score(model=model)
    # macs , params2 = get_model_complexity_info(model, (3, 512, 512), print_per_layer_stat=False, verbose=False)
    mon.console.log(f"Params: {params:.4f}")
    mon.console.log(f"FLOPs : {flops:.4f}")
    # mon.console.log(f"MACs  : {macs:.4f}")


def load_model(model, path_weights):
    checkpoints = torch.load(str(path_weights), map_location="cpu", weights_only=False)
    weights     = checkpoints["params"]
    # weights     = {"module." + key: value for key, value in weights.items()}
    model.load_state_dict(weights)
    # print("Loaded weights correctly")
    return model


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
    model, _, _ = create_model(args["network"], rank=0, device=device, torchrun=args.torchrun)
    model = load_model(model, path_weights=pretrained)
    model.eval()
    
    # Benchmark
    if args.benchmark:
        benchmark(model)
    
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
            if args.resize and (h0 >= 1500 or w0 >= 1500):
                new_size   = [int(dim // 2) for dim in (h0, w0)]
                downsample = torchvision.transforms.Resize(new_size)
            else:
                downsample = torch.nn.Identity()
            image  = downsample(image)
            image  = image.to(device)
            timers.preprocess.tock()

            # Infer
            timers.infer.tick()
            outputs = model(image, side_loss=False)
            timers.infer.tock()

            # Postprocess
            timers.postprocess.tick()
            if args.resize:
                upsample = torchvision.transforms.Resize((h0, w0))
            else:
                upsample = torch.nn.Identity()
            enhanced = upsample(outputs)
            enhanced = torch.clamp(enhanced, 0.0, 1.0)
            enhanced = enhanced[:, :, :h0, :w0]
            timers.postprocess.tock()

            # Save
            if args.save_image:
                out_dir  = mon.parse_output_dir(args.save_dir, data_name, mon.SAVE_IMAGE_DIR, path, args.keep_subdirs, args.save_nearby)
                out_path = out_dir / f"{path.stem}{mon.SAVE_IMAGE_EXT}"
                mon.save_image(enhanced, out_path)
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
