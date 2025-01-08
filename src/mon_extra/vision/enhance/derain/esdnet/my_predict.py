#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import copy
import math

import torch
import torch.optim
import torchvision

import mon
from model import model
from spikingjelly.activation_based import functional

console      = mon.console
current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# region Predict

def get_scoremap(b, c, h, w, is_mean: bool = True) -> torch.Tensor:
    center_h = h / 2
    center_w = w / 2
    score    = torch.ones((b, c, h, w))
    if not is_mean:
        for h in range(h):
            for w in range(w):
                score[:, :, h, w] = 1.0 / (math.sqrt((h - center_h) ** 2 + (w - center_w) ** 2 + 1e-3))
    return score


def split_image(
    img_tensor  : torch.Tensor,
    crop_size   : int = 80,
    overlap_size: int = 8
) -> tuple[list, list]:
    b, c, h, w = img_tensor.shape
    
    h_starts = [x for x in range(0, h, crop_size - overlap_size)]
    while h_starts[-1] + crop_size >= h:
        h_starts.pop()
    h_starts.append(h - crop_size)
    
    w_starts = [x for x in range(0, w, crop_size - overlap_size)]
    while w_starts[-1] + crop_size >= w:
        w_starts.pop()
    w_starts.append(w - crop_size)
   
    starts     = []
    split_data = []
    for hs in h_starts:
        for ws in w_starts:
            c_img_data = img_tensor[:, :, hs:hs + crop_size, ws:ws + crop_size]
            starts.append((hs, ws))
            split_data.append(c_img_data)
    return split_data, starts


def merge_image(split_data, starts, resolution=(1, 3, 80, 80)) -> torch.Tensor:
    b, c, h, w = resolution[0], resolution[1], resolution[2], resolution[3]
    tot_score  = torch.zeros((b, c, h, w))
    merge_img  = torch.zeros((b, c, h, w))
    scoremap   = get_scoremap(b, c, h, w, is_mean=False)
    for simg, cstart in zip(split_data, starts):
        hs, ws = cstart
        merge_img[:, :, hs:hs + h, ws:ws + w] += scoremap * simg
        tot_score[:, :, hs:hs + h, ws:ws + w] += scoremap
    merge_img = merge_img / tot_score
    return merge_img


def predict(args: argparse.Namespace):
    # General config
    data         = args.data
    save_dir     = args.save_dir
    weights      = args.weights
    device       = mon.set_device(args.device)
    imgsz        = args.imgsz
    resize       = args.resize
    crop_size    = 80
    overlap_size = 8
    benchmark    = args.benchmark
    save_image   = args.save_image
    save_debug   = args.save_debug
    use_fullpath = args.use_fullpath
    
    # Model
    model_restoration = model.to(device)
    functional.set_step_mode(model_restoration, step_mode="m")
    functional.set_backend(model_restoration,   backend="cupy")
    model_restoration.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    model_restoration.to(device)
    model_restoration.eval()
    
    # Benchmark
    if benchmark:
        flops, params, avg_time = mon.compute_efficiency_score(
            model      = copy.deepcopy(model_restoration),
            image_size = imgsz,
            channels   = 3,
            runs       = 1000,
            use_cuda   = True,
            verbose    = False,
        )
        console.log(f"FLOPs  = {flops:.4f}")
        console.log(f"Params = {params:.4f}")
        console.log(f"Time   = {avg_time:.17f}")
    
    # Data I/O
    console.log(f"[bold red]{data}")
    data_name, data_loader, data_writer = mon.parse_io_worker(
        src         = data,
        dst         = save_dir,
        to_tensor   = False,
        denormalize = True,
        verbose     = False,
    )
    
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
                image      = datapoint.get("image")
                image      = image.to(device)
                meta       = datapoint.get("meta")
                image_path = mon.Path(meta["path"])
                b, c, h, w = image.shape
                
                # Infer
                timer.tick()
                split_data, starts = split_image(image, crop_size=crop_size, overlap_size=overlap_size)
                for j, data in enumerate(split_data):
                    split_data[j] = model_restoration(data).to(device)
                    functional.reset_net(model_restoration)
                    split_data[j] = split_data[j].cpu()
                enhanced = merge_image(split_data, starts, resolution=(b, c, h, w))
                enhanced = torch.clamp(enhanced, 0, 1).permute(0, 2, 3, 1).numpy()
                timer.tock()
                
                # Save
                if save_image:
                    if use_fullpath:
                        rel_path    = image_path.relative_path(data_name)
                        output_path = save_dir / rel_path.parent / image_path.name
                    else:
                        output_path = save_dir / data_name / image_path.name
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    torchvision.utils.save_image(enhanced, str(output_path))
        
        avg_time = float(timer.avg_time)
        console.log(f"Average time: {avg_time}")

# endregion


# region Main

def main() -> str:
    args = mon.parse_predict_args(model_root=current_dir)
    predict(args)


if __name__ == "__main__":
    main()
    
# endregion
