#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements GCE-Net model prediction pipeline for low-light image enhancement."""

import box
import cv2
import torch
from torch import inference_mode

import mon
from mon import albumentations as A
from mon.vision.enhance.lle import gcenet

mon.dev()

current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# ----- Predict -----
@torch.no_grad()
def predict(args: dict | box.Box) -> str:
    # Start
    mon.rt.print_run_summary(args)

    # Device
    device = mon.create_device(args.device)

    # Seed
    mon.set_random_seed(args.seed)

    # Pretrained
    pretrained = args.resume
    if args.weights and args.weights.is_weights_file(exist=True):
        pretrained = args.weights
    if pretrained and pretrained.is_weights_file(exist=True):
        mon.log(f"Pretrained: {pretrained}.")
    else:
        raise ValueError(f"Invalid weights file: {pretrained}.")

    # Model
    # iters = args["network"]["iters"]
    iters = 6
    args.network |= {
        "iters"         : iters,
        "use_depth"     : True,
        "inference_mode": True,
    }
    # model = gcenet.GCENet_MO(iters=iters, use_depth=True, inference_mode=True)
    model = mon.MODELS.build(args.model, **args.network)
    model.load_state_dict(torch.load(pretrained, weights_only=True))
    model = model.to(device)
    model.eval()
    
    # Benchmark
    if args.benchmark:
        mon.nn.benchmark(model)
   
    # Data I/O
    imgsz     = args.imgsz if args.resize else (0, 0)
    transform = A.Compose([
        A.ResizeDivisibleBy(height=imgsz[0], width=imgsz[1], divisor=32),
        A.Normalize(normalization="min_max"),
        A.ToTensorV2(transpose_mask=True),
    ])
    data_name, dataloader = mon.data.build_dataloader(args.data, args.root, transform)

    # Predict
    timers = mon.TimeProfiler()
    timers.total.tick()
    with mon.create_progress_bar() as pbar:
        for i, datapoint in pbar.track(
            sequence    = enumerate(dataloader),
            total       = len(dataloader),
            description = f"[bright_yellow]Predicting"
        ):
            # Preprocess
            timers.preprocess.tick()
            meta   = datapoint["meta"][0]
            path   = mon.Path(meta["path"])
            h0, w0 = mon.image.imgsz(meta["orig_shape"])
            image  = datapoint["image"]
            image  = image.to(device)
            depth  = datapoint.get("depth", None)
            depth  = depth.to(device) if depth is not None else None
            timers.preprocess.tock()

            # Infer
            timers.infer.tick()
            outputs = model(image, depth)
            timers.infer.tock()

            # Postprocess
            timers.postprocess.tick()
            enhanced = outputs[-1]
            enhanced = mon.image.to_array(enhanced)
            h1, w1   = mon.image.imgsz(enhanced)
            if (h1, w1) != (h0, w0):
                enhanced = cv2.resize(enhanced, (w0, h0))
            timers.postprocess.tock()

            # Save
            if args.save_image:
                out_dir  = mon.rt.parse_output_dir(args.save_dir, data_name, mon.SAVE_IMAGE_DIR, path, args.keep_subdirs, args.save_nearby)
                out_path = out_dir / f"{path.stem}{mon.SAVE_IMAGE_EXT}"
                mon.image.save_image(enhanced, out_path)
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
