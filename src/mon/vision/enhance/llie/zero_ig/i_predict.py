#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import logging
import sys

import cv2
import numpy as np
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils
from PIL import Image
from thop import profile
from torch.autograd import Variable

import mon
from model import *

console      = mon.console
current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# region Predict

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


def predict(args: argparse.Namespace):
    # Parse args
    hostname     = args.hostname
    data         = args.data
    fullname     = args.fullname
    save_dir     = args.save_dir
    weights      = args.weights
    device       = args.device
    seed         = args.seed
    imgsz        = args.imgsz
    resize       = args.resize
    epochs       = args.epochs
    steps        = args.steps
    benchmark    = args.benchmark
    save_image   = args.save_image
    save_debug   = args.save_debug
    use_fullpath = args.use_fullpath
    verbose      = args.verbose
    
    # Start
    console.rule(f"[bold red] {fullname}")
    console.log(f"Machine: {hostname}")
    
    # Device
    device = mon.set_device(device)
    if torch.cuda.is_available():
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        cudnn.benchmark = True
        cudnn.enabled   = True
    else:
        torch.set_default_tensor_type('torch.FloatTensor')
        logging.info('no gpu device available')
        sys.exit(1)
    
    # Seed
    mon.set_random_seed(seed)
    
    # Data I/O
    console.log(f"[bold red]{data}")
    data_name, data_loader, data_writer = mon.parse_io_worker(
        src         = data,
        dst         = save_dir,
        to_tensor   = False,
        denormalize = True,
        verbose     = False,
    )
    
    # Benchmark
    if benchmark:
        model = Network()
        mon.compute_efficiency_score(model=model, image_size=imgsz, channels=3, verbose=True)
        total_params = calculate_model_parameters(model)
        console.log(f"Total Params = {total_params:.4f}")
        
    # Predicting
    timer = mon.Timer()
    with mon.get_progress_bar() as pbar:
        for i, datapoint in pbar.track(
            sequence    = enumerate(data_loader),
            total       = len(data_loader),
            description = f"[bright_yellow] Predicting"
        ):
            # Input
            meta       = datapoint.get("meta")
            image_path = mon.Path(meta["path"])
            image      = Image.open(image_path).convert("RGB")
            image      = (np.asarray(image) / 255.0)
            image      = torch.from_numpy(image).float()
            image      = image.permute(2, 0, 1)
            image      = image.unsqueeze(0).to(device)
            
            # Optimize
            timer.tick()
            model = Network()
            model.enhance.in_conv.apply(model.enhance_weights_init)
            model.enhance.conv.apply(model.enhance_weights_init)
            model.enhance.out_conv.apply(model.enhance_weights_init)
            model = model.to(device)
            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=3e-4)
            input     = Variable(image, requires_grad=False).to(device)
            for _ in range(epochs):
                optimizer.zero_grad()
                optimizer.param_groups[0]["capturable"] = True
                loss = model._loss(input)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
            model = Finetunemodel(model.state_dict())
            input = Variable(image, volatile=True).to(device)
            enhance, output = model(input)
            timer.tock()
            
            # Post-processing
            enhance = save_images(enhance)
            output  = save_images(output)
            enhance = cv2.cvtColor(enhance, cv2.COLOR_BGR2RGB)
            output  = cv2.cvtColor(output,  cv2.COLOR_BGR2RGB)
            
            # Save
            if save_image:
                if use_fullpath:
                    rel_path   = image_path.relative_path(data_name)
                    output_dir = save_dir / rel_path.parents[0]
                else:
                    output_dir = save_dir / data_name
                output_path    = output_dir / f"{image_path.stem}.jpg"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(output_path), enhance)
            if save_debug:
                if use_fullpath:
                    rel_path   = image_path.relative_path(data_name)
                    debug_dir  = save_dir / rel_path.parents[1] / f"{rel_path.parent.name}_denoise"
                else:
                    debug_dir  = save_dir / f"{data_name}_denoise"
                output_path    = debug_dir / f"{image_path.stem}.jpg"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(output_path), output)
    
    # Finish
    console.log(f"Average time: {float(timer.avg_time)}")

# endregion


# region Main

def main() -> str:
    args = mon.parse_predict_args(model_root=current_dir)
    predict(args)


if __name__ == "__main__":
    main()
    
# endregion
