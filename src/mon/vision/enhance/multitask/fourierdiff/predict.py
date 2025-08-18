#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements FourierDiff model for zero-shot joint low-light enhancement and deblurring.

References:
    - Paper: "Fourier Priors-Guided Diffusion for Zero-Shot Joint Low-Light
      Enhancement and Deblurring," CVPR 2024.
    - Code: https://github.com/aipixel/FourierDiff
"""

import argparse

import box
import torch
import yaml

import albumentations as A
import box
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

import mon
from mon import console, metrics, Path, tfms, optims
from mon.vision.enhance.multitask import fourierdiff

torch.set_printoptions(sci_mode=False)

current_file = Path(__file__).absolute()
current_dir  = current_file.parents[0]


# ----- Utils -----
def benchmark(model: nn.Module):
    flops, params = mon.metric.compute_complexity(model=model)
    mon.log(f"Params    : {params:.4f}")
    mon.log(f"FLOPs     : {flops:.4f}")


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


# ----- Predict -----
@torch.no_grad()
def predict(args: dict | box.Box) -> str:
    cfg_path = current_dir / "fourierdiff" / "option" / args.cfg
    with open(str(cfg_path), "r") as f:
        cfg = yaml.safe_load(f)
    cfg = dict2namespace(cfg)

    # Start
    mon.rt.print_run_summary(args)
    
    # Device
    device     = mon.create_device(args.device)
    cfg.device = device

    # Seed
    mon.set_random_seed(args.seed)

    # Data I/O
    data_name, data_loader = mon.parse_data_loader(args.data, args.root, True, verbose=False)

    # Pretrained
    pretrained = args.resume
    if args.weights and args.weights.is_weights_file(exist=True):
        pretrained = args.weights
    if pretrained and pretrained.is_weights_file(exist=True):
        mon.log(f"Pretrained: {pretrained}.")
    else:
        raise ValueError(f"Invalid weights file: {pretrained}.")

    # Model
    runner = fourierdiff.FourierDiff(dict2namespace(args), cfg)

    # Benchmark
    # if args.benchmark:
    #     benchmark(model)
    
    # Predict
    timers = mon.TimeProfiler()
    timers.total.tick()
    runner.sample(
        pretrained,
        data_name,
        data_loader,
        args.imgsz,
        args.resize,
        args.save_image,
        args.save_dir,
        args.keep_subdirs,
        args.save_nearby,
        timers
    )
    timers.total.tock()

    # Finish
    timers.print()
    return str(args.save_dir)


# ----- Main -----
def main() -> str:
    args = mon.rt.parse_predict_args(model_root=current_dir)
    predict(args)


if __name__ == "__main__":
    main()
