#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import random
from typing import Sequence

import torch.optim
from PIL import Image

import mon
from data.base_dataset import get_transform
from models import create_model
from options.test_options import TestOptions
from util import util

console      = mon.console
current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# region Predict

def predict(args: argparse.Namespace):
    # Parse args
    hostname     = args.hostname
    root         = args.root
    data         = args.data
    fullname     = args.fullname
    save_dir     = args.save_dir
    weights      = args.weights
    device       = args.device
    seed         = args.seed
    imgsz        = args.imgsz
    imgsz        = imgsz[0] if isinstance(imgsz, Sequence) else imgsz
    resize       = args.resize
    epochs       = args.epochs
    steps        = args.steps
    benchmark    = args.benchmark
    save_image   = args.save_image
    save_debug   = args.save_debug
    use_fullpath = args.use_fullpath
    verbose      = args.verbose
    
    # Hard-code some parameters for test
    opt                = TestOptions().parse()  # get test options
    opt.num_threads    = 0       # test code only supports num_threads = 0
    opt.batch_size     = 1       # test code only supports batch_size  = 1
    opt.serial_batches = True    # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip        = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id     = -1      # no visdom display; the test code saves the results to a HTML file.
    opt.device         = device
    
    # Start
    console.rule(f"[bold red] {fullname}")
    console.log(f"Machine: {hostname}")
    
    # Device
    device = mon.set_device(device)
    
    # Seed
    mon.set_random_seed(seed)
    
    # Data I/O
    console.log(f"[bold red]{data}")
    data_name, data_loader, data_writer = mon.parse_io_worker(
        src         = data,
        dst         = save_dir,
        to_tensor   = True,
        denormalize = True,
        verbose     = False,
    )
    testB_dir   = current_dir / "dataset" / "testB"
    testB_files = sorted([f for f in testB_dir.glob("*") if f.is_image_file()])
    testB_size  = len(testB_files)
    transform_A = get_transform(opt)
    transform_B = get_transform(opt)
    
    # Model
    model = create_model(opt)    # create a model given opt.model and other options
    model.setup(weights, opt)    # regular setup: load and print networks; create schedulers
    model = model.to(device)
    if opt.eval:
        model.eval()
    
    # Benchmark
    if benchmark:
        flops, params = model.measure_efficiency_score()
        console.log(f"FLOPs : {flops:.4f}")
        console.log(f"Params: {params:.4f}")
    
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
                meta       = datapoint.get("meta")
                image_path = mon.Path(meta["path"])
                indexB     = random.randint(0, testB_size - 1)
                imageA     = Image.open(image_path).convert("RGB")
                imageB     = Image.open(testB_files[indexB]).convert("RGB")
                w0, h0     = imageA.size
                imageA     = transform_A(imageA).unsqueeze(0).to(device)
                imageB     = transform_B(imageB).unsqueeze(0).to(device)
                dp = {
                    "A"      : imageA,
                    "B"      : imageB,
                    "A_paths": image_path,
                    "B_paths": testB_files[indexB]
                }
                
                # Infer
                timer.tick()
                model.set_input(dp)
                model.test()
                visuals = model.get_current_visuals()
                fake_B  = visuals.get("fake_B")
                timer.tock()
                
                # Post-process
                h1, w1 = mon.get_image_size(fake_B)
                if h1 != h0 or w1 != w0:
                    fake_B = mon.resize(fake_B, (h0, w0))
                fake_B = util.tensor2im(fake_B)
                
                # Save
                if save_image:
                    if use_fullpath:
                        rel_path    = image_path.relative_path(data_name)
                        output_path = save_dir / rel_path.parent / f"{image_path.stem}.jpg"
                    else:
                        output_path = save_dir / data_name / f"{image_path.stem}.jpg"
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    image_pil = Image.fromarray(fake_B)
                    image_pil.save(str(output_path))
                '''
                if save_debug:
                    if use_fullpath:
                        rel_path    = image_path.relative_path(data_name)
                        output_path = save_dir / f"{rel_path.parent}_debug"
                    else:
                        output_path = save_dir / f"{rel_path.parent}_debug"
                    output_path.mkdir(parents=True, exist_ok=True)
                    # torchvision.utils.save_image(g_a, str(output_path / f"{image_path.stem}_g_a.jpg"))
                    # torchvision.utils.save_image(pre, str(output_path / f"{image_path.stem}_pre.jpg"))
                '''
                
    console.log(f"Average time: {timer.avg_time}")

# endregion


# region Main

def main() -> str:
    args = mon.parse_predict_args(model_root=current_dir)
    predict(args)


if __name__ == "__main__":
    main()
    
# endregion
