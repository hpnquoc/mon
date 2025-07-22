#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements the paper: "Implicit Neural Representation for Cooperative Low-light
Image Enhancement," ICCV 2023.

References:
    - https://github.com/Ysz2022/NeRCo
"""

import random

import box
import torch.optim
from PIL import Image

import mon
from data.base_dataset import get_transform
from models import create_model
from options.test_options import TestOptions
from util import util

current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# ----- Utils -----
def benchmark(model: torch.nn.Module):
    flops, params = model.compute_efficiency_score()
    mon.console.log(f"Params: {params:.4f}")
    mon.console.log(f"FLOPs : {flops:.4f}")


# ----- Predict -----
@torch.no_grad()
def predict(args: dict | box.Box) -> str:
    # Hard-code some parameters for test
    cfgs                = TestOptions().parse()  # get test options
    cfgs.num_threads    = 0       # test code only supports num_threads = 0
    cfgs.batch_size     = 1       # test code only supports batch_size  = 1
    cfgs.serial_batches = True    # disable data shuffling; comment this line if results on randomly chosen images are needed.
    cfgs.no_flip        = True    # no flip; comment this line if results on flipped images are needed.
    cfgs.display_id     = -1      # no visdom display; the test code saves the results to a HTML file.
    
    # Start
    mon.print_run_summary(args)

    # Device
    device = mon.set_device(args.device)
    cfgs.device = device
    
    # Seed
    mon.set_random_seed(args.seed)

    # Data I/O
    data_name, data_loader = mon.parse_data_loader(args.data, args.root, True, verbose=False)

    testB_dir   = current_dir / "dataset" / "testB"
    testB_files = sorted([f for f in testB_dir.glob("*") if f.is_image_file()])
    testB_size  = len(testB_files)
    transform_A = get_transform(cfgs)
    transform_B = get_transform(cfgs)
    
    # Pretrained
    pretrained = args.weights
    '''
    if args.weights and args.weights.is_weights_file(exist=True):
        pretrained = args.weights
    if pretrained and pretrained.is_weights_file(exist=True):
        mon.console.log(f"Pretrained: {pretrained}.")
    else:
        mon.console.log(f"Pretrained: {None}, training from scratch.")
    '''

    # Model
    model = create_model(cfgs)          # create a model given opt.model and other options
    model.setup(pretrained, cfgs)       # regular setup: load and print networks; create schedulers
    model = model.to(device)
    if cfgs.eval:
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
            indexB = random.randint(0, testB_size - 1)
            imageA = Image.open(path).convert("RGB")
            imageB = Image.open(testB_files[indexB]).convert("RGB")
            w0, h0 = imageA.size
            imageA = transform_A(imageA).unsqueeze(0).to(device)
            imageB = transform_B(imageB).unsqueeze(0).to(device)
            dp     = {
                "A"      : imageA,
                "B"      : imageB,
                "A_paths": path,
                "B_paths": testB_files[indexB]
            }
            timers.preprocess.tock()

            # Infer
            timers.infer.tick()
            model.set_input(dp)
            model.test()
            timers.infer.tock()

            # Postprocess
            timers.postprocess.tick()
            outputs  = model.get_current_visuals()
            enhanced = outputs.get("fake_B")
            h1, w1 = mon.image_size(enhanced)
            if h1 != h0 or w1 != w0:
                enhanced = mon.resize(enhanced, (h0, w0))
            enhanced = util.tensor2im(enhanced)
            timers.postprocess.tock()

            # Save
            if args.save_image:
                out_dir  = mon.parse_output_dir(args.save_dir, data_name, mon.SAVE_IMAGE_DIR, path, args.keep_subdirs, args.save_nearby)
                out_path = out_dir / f"{path.stem}{mon.SAVE_IMAGE_EXT}"
                mon.save_image(enhanced, out_path)

            '''
                if save_debug:
                    if keep_subdirs:
                        rel_path    = image_path.relative_path(data_name)
                        output_path = save_dir / f"{rel_path.parent}_debug"
                    else:
                        output_path = save_dir / f"{rel_path.parent}_debug"
                    output_path.mkdir(parents=True, exist_ok=True)
                    # torchvision.utils.save_image(g_a, str(output_path / f"{image_path.stem}_g_a.jpg"))
                    # torchvision.utils.save_image(pre, str(output_path / f"{image_path.stem}_pre.jpg"))
                '''
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
