#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements the paper: "Deep Retinex Decomposition for Low-Light Enhancement," BMCV 2018.

References:
    - https://github.com/aasharma90/RetinexNet_PyTorch
"""

import box

import mon
from model import RetinexNet

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
    model = RetinexNet(args.imgsz, args.benchmark)
    model = model.to(device)
    
    # Predict
    timers = mon.TimeProfiler()
    with mon.create_progress_bar() as pbar:
        for i, datapoint in pbar.track(
            sequence    = enumerate(data_loader),
            total       = len(data_loader),
            description = f"[bright_yellow]Listing images",
        ):
            # Preprocess
            timers.preprocess.tick()
            path = mon.Path(datapoint["meta"]["path"])
            timers.preprocess.tock()

            out_dir = mon.parse_output_dir(args.save_dir, data_name, mon.SAVE_IMAGE_DIR, path, args.keep_subdirs, args.save_nearby)
            # out_dir = out_dir / mon.SAVE_IMAGE_DIR
            out_dir.mkdir(parents=True, exist_ok=True)

            # Infer
            timers.infer.tick()
            model.predict(
                [path],
                res_dir  = str(out_dir),
                ckpt_dir = str(args.weights),
                imgsz    = args.imgsz,
                resize   = args.resize
            )
            timers.infer.tock()
    
    # Finish
    timers.print()
    return str(args.save_dir)


# ----- Main -----
def main() -> str:
    args = mon.parse_predict_args(model_root=current_dir)
    predict(args)


if __name__ == "__main__":
    main()
