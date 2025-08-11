#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""RUAS model prediction pipeline for low-light image enhancement.

References:
    - Paper: "Retinex-inspired Unrolling with Cooperative Prior Architecture
      Search for Low-light Image Enhancement," 2021.
    - Code: https://github.com/KarelZhang/RUAS
"""

import box
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.utils
from PIL import Image

import mon
from mon.vision.enhance.lle import ruas

current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# ----- Utils -----
def benchmark(model: torch.nn.Module):
    flops, params = mon.compute_efficiency_score(model=model)
    mon.console.log(f"Params    : {params:.4f}")
    mon.console.log(f"FLOPs     : {flops:.4f}")


def save_images(tensor, path):
    image_numpy = tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)))
    im = Image.fromarray(np.clip(image_numpy * 255.0, 0, 255.0).astype("uint8"))
    im.save(path, 'png')


# ----- Predict -----
@torch.no_grad()
def predict(args: dict | box.Box) -> str:
    # Start
    mon.print_run_summary(args)

    # Device
    device          = mon.set_device(args.device)
    cudnn.benchmark = True
    cudnn.enabled   = True
    
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
    model = ruas.RUAS()
    model.load_state_dict(torch.load(str(pretrained), weights_only=True))
    for p in model.parameters():
        p.requires_grad = False
    model = model.to(device)
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
            if args.resize and (h0 != args.imgsz[0] or w0 != args.imgsz[1]):
                image = mon.resize(image, size=args.imgsz)
            image  = image.to(device)
            timers.preprocess.tock()
            
            # Infer
            timers.infer.tick()
            outputs = model(image)
            timers.infer.tock()

            # Postprocess
            timers.postprocess.tick()
            u_list, _ = outputs
            if args.resize and (h0 != args.imgsz[0] or w0 != args.imgsz[1]):
                u_list = [mon.resize(u, size=(h0, w0)) for u in u_list]
            timers.postprocess.tock()

            # Save
            if args.save_image:
                out_dir  = mon.parse_output_dir(args.save_dir, data_name, mon.SAVE_IMAGE_DIR, path, args.keep_subdirs, args.save_nearby)
                out_path = out_dir / f"{path.stem}{mon.SAVE_IMAGE_EXT}"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                save_images(u_list[-1], str(out_path))
                # save_images(u_list[-1], str(args.output_dir / "lol" / u_name))
                # save_images(u_list[-2], str(args.output_dir / "dark" / u_name))
                """
                if args.model == "lol":
                    save_images(u_list[-1], u_path)
                elif args.model == "upe" or args.model == "dark":
                    save_images(u_list[-2], u_path)
                """

            if args.save_debug:
                out_dir  = mon.parse_output_dir(args.save_dir, data_name, mon.SAVE_DEBUG_DIR, path, args.keep_subdirs, args.save_nearby)
                out_path = out_dir / f"{path.stem}_dark{mon.SAVE_IMAGE_EXT}"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                save_images(u_list[-2], str(out_path))
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
