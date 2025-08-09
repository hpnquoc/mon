#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""SNR model prediction pipeline for low-light image enhancement.

References:
    - Paper: "SNR-aware Low-Light Image Enhancement," CVPR 2022.
    - Code: https://github.com/dvlab-research/SNR-Aware-Low-Light-Enhance
"""

import os
import sys

import box
import cv2
import numpy as np
import torch

import mon

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from snr.data import util as dutil
from snr.options import options as option
from snr.utils import util as util
from snr.models import create_model

current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# ----- Utils -----
def benchmark(model: torch.nn.Module):
    return
    flops, params = mon.compute_efficiency_score(model=model)
    mon.console.log(f"Params    : {params:.4f}")
    mon.console.log(f"FLOPs     : {flops:.4f}")


# ----- Predict -----
@torch.no_grad()
def predict(args: dict | box.Box) -> str:
    cfg_path = current_dir / "snr" / "options" / "test" / args.cfg
    cfgs     = option.parse(str(cfg_path), is_train=False)
    cfgs     = option.dict_to_nonedict(cfgs)
    
    # Start
    mon.print_run_summary(args)

    # Device
    device         = mon.set_device(args.device)
    cfgs["device"] = device

    # Seed
    mon.set_random_seed(args.seed)

    # Data I/O
    data_name, data_loader = mon.parse_data_loader(args.data, args.root, False, verbose=False)

    # Pretrained
    pretrained = args.resume
    if args.weights and args.weights.is_weights_file(exist=True):
        pretrained = args.weights
    if pretrained and pretrained.is_weights_file(exist=True):
        mon.console.log(f"Pretrained: {pretrained}.")
    else:
        raise ValueError(f"Invalid weights file: {pretrained}.")
    cfgs["path"]["pretrain_model_G"] = str(pretrained)

    # Model
    model = create_model(cfgs)
    
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
            image  = dutil.read_img(None, str(path))
            h0, w0 = mon.image_size(image)
            if args.resize and (h0 != args.imgsz[0] or w0 != args.imgsz[1]):
                image = mon.resize(image, size=args.imgsz)
            else:
                image = mon.resize(image, divisible_by=32)
            image_nf = cv2.blur(image, (5, 5))
            image_nf = image_nf * 1.0 / 255.0
            image_nf = torch.from_numpy(np.ascontiguousarray(np.transpose(image_nf, (2, 0, 1)))).float()
            image    = torch.from_numpy(np.ascontiguousarray(np.transpose(image,    (2, 0, 1)))).float()
            image    = image.unsqueeze(0).to(device)
            image_nf = image_nf.unsqueeze(0).to(device)
            timers.preprocess.tock()

            # Infer
            timers.infer.tick()
            model.feed_data(
                data = {
                    "idx": i,
                    "LQs": image,
                    "nf" : image_nf,
                },
                need_GT=False
            )
            model.test()
            timers.infer.tock()

            # Postprocess
            timers.postprocess.tick()
            outputs  = model.get_current_visuals(need_GT=False)
            enhanced = util.tensor2img(outputs["rlt"])  # uint8
            enhanced = cv2.resize(enhanced, (w0, h0))
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
