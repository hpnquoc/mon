#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""LIME model prediction pipeline for low-light image enhancement.

References:
    - Paper: "LIME: Low-light Image Enhancement via Illumination Map Estimation,"
      TIP 2006.
    - Code: https://github.com/pvnieo/Low-light-Image-Enhancement
"""

import os
import sys

import box
import cv2

import mon

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from model import *

current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# ----- Predict -----
def predict(args: dict | box.Box) -> str:
    # Start
    mon.print_run_summary(args)

    # Device
    device = mon.set_device(args.device)

    # Seed
    mon.set_random_seed(args.seed)

    # Data I/O
    data_name, data_loader = mon.parse_data_loader(args.data, args.root, False, verbose=False)
    
    # Model
    model = LIME(
        gamma   = args["network"]["gamma"],
        lambda_ = args["network"]["lambda_"],
        dual    = not args["network"]["lime"],
        sigma   = args["network"]["sigma"],
        bc      = args["network"]["bc"],
        bs      = args["network"]["bs"],
        be      = args["network"]["be"],
        eps     = args["network"]["eps"],
    )
    
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
            if args.resize and (h0 != args.imgsz[0] or w0 != args.imgsz[1]):
                image = mon.resize(image, size=args.imgsz)
            timers.preprocess.tock()

            # Infer
            timers.infer.tick()
            outputs = model(image)
            timers.infer.tock()

            # Postprocess
            timers.postprocess.tick()
            enhanced = outputs
            if args.resize and (h0 != args.imgsz[0] or w0 != args.imgsz[1]):
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
