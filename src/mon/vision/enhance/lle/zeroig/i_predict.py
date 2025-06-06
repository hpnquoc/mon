#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements the paper: "Zero-Shot Illumination-Guided Joint Denoising and Adaptive
Enhancement for Low-Light Images," CVPR 2024.

References:
    - https://github.com/Doyle59217/ZeroIG
"""

import logging
import sys

import box
import cv2
import numpy as np
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils
from thop import profile
from torch.autograd import Variable

import mon
from model import *

current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# ----- Utils -----
def save_images(tensor):
    image_numpy = tensor[0].detach().cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)))
    im          = np.clip(image_numpy * 255.0, 0, 255.0).astype("uint8")
    return im


def calculate_model_parameters(model):
    return sum(p.numel() for p in model.parameters())


def calculate_model_flops(model, input_tensor):
    flops, _           = profile(model, inputs=(input_tensor,))
    flops_in_gigaflops = flops / 1e9  # Convert FLOPs to gigaflops (G)
    return flops_in_gigaflops


def benchmark():
    model = Network()
    # flops, params = mon.compute_efficiency_score(model=model)
    total_params  = calculate_model_parameters(model)
    # mon.console.log(f"FLOPs : {flops:.4f}")
    # mon.console.log(f"Params: {params:.4f}")
    mon.console.log(f"Total Params = {total_params:.4f}")


# ----- Predict -----
@torch.no_grad()
def predict(args: dict | box.Box) -> str:
    # Start
    mon.print_run_summary(args)

    # Device
    device = mon.set_device(args.device)
    if torch.cuda.is_available():
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        cudnn.benchmark = True
        cudnn.enabled   = True
    else:
        torch.set_default_tensor_type("torch.FloatTensor")
        logging.info("no gpu device available")
        sys.exit(1)

    # Seed
    mon.set_random_seed(args.seed)

    # Data I/O
    data_name, data_loader = mon.parse_data_loader(args.data, args.root, True, verbose=False)

    # Benchmark
    # Benchmark
    if args.benchmark:
        benchmark()
        
    # Predict
    timers = mon.TimeProfiler()
    with mon.create_progress_bar() as pbar:
        for i, datapoint in pbar.track(
            sequence    = enumerate(data_loader),
            total       = len(data_loader),
            description = f"[bright_yellow] Predicting"
        ):
            # Preprocess
            timers.preprocess.tick()
            path   = mon.Path(datapoint["meta"]["path"])
            image  = datapoint["image"]
            timers.preprocess.tock()

            # Optimize
            timers.infer.tick()
            model = Network()
            model.enhance.in_conv.apply(model.enhance_weights_init)
            model.enhance.conv.apply(model.enhance_weights_init)
            model.enhance.out_conv.apply(model.enhance_weights_init)
            model = model.to(device)
            model.train()
            optimizer = torch.optim.Adam(model.parameters(), **args.optimizer)
            input     = Variable(image, requires_grad=False).to(device)
            for _ in range(args.epochs):
                optimizer.zero_grad()
                optimizer.param_groups[0]["capturable"] = True
                loss = model._loss(input)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
            model = Finetunemodel(model.state_dict())
            input = Variable(image).to(device)
            outputs = model(input)
            timers.infer.tock()
            
            # Postprocess
            timers.postprocess.tick()
            enhanced, denoise = outputs
            enhanced = save_images(enhanced)
            denoise  = save_images(denoise)
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
            denoise  = cv2.cvtColor(denoise,  cv2.COLOR_BGR2RGB)
            denoise  = cv2.cvtColor(denoise,  cv2.COLOR_BGR2RGB)
            timers.postprocess.tock()

            # Save
            if args.save_image:
                out_dir  = mon.parse_output_dir(args.save_dir, data_name, mon.SAVE_IMAGE_DIR, path, args.keep_subdirs, args.save_nearby)
                out_path = out_dir / f"{path.stem}{mon.SAVE_IMAGE_EXT}"
                mon.save_image(enhanced, out_path)

            if args.save_debug:
                out_dir  = mon.parse_output_dir(args.save_dir, data_name, f"{mon.SAVE_IMAGE_DIR}_denoise", path, args.keep_subdirs, args.save_nearby)
                out_path = out_dir / f"{path.stem}{mon.SAVE_IMAGE_EXT}"
                mon.save_image(denoise, out_path)
    
    # Finish
    timers.print()
    return str(args.save_dir)


# ----- Main -----
def main() -> str:
    args = mon.parse_predict_args(model_root=current_dir)
    predict(args)


if __name__ == "__main__":
    main()
