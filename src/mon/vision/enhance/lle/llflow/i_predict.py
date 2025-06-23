#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements the paper: "Low-Light Image Enhancement with Normalizing Flow," AAAI 2022.

References:
    - https://github.com/wyf0912/LLFlow
"""

import os

import box
import cv2
import numpy as np
import torch

import mon
import options.options as option
from models import create_model

current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# ----- Utils -----
def benchmark(model: torch.nn.Module):
    flops, params = model.compute_efficiency_score()
    mon.console.log(f"Params: {params:.4f}")
    mon.console.log(f"FLOPs : {flops:.4f}")


def t(array):
    return torch.Tensor(np.expand_dims(array.transpose([2, 0, 1]), axis=0).astype(np.float32)) / 255


def rgb(t):
    return (np.clip((t[0] if len(t.shape) == 4 else t).detach().cpu().numpy().transpose([1, 2, 0]), 0, 1) * 255).astype(np.uint8)


def imread(path):
    return cv2.imread(path)[:, :, [2, 1, 0]]


def imwrite(path, img):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img[:, :, [2, 1, 0]])


def imCropCenter(img, size):
    h, w, c = img.shape
    h_start = max(h // 2 - size // 2, 0)
    h_end   = min(h_start + size, h)
    w_start = max(w // 2 - size // 2, 0)
    w_end   = min(w_start + size, w)
    return img[h_start:h_end, w_start:w_end]


def impad(img, top=0, bottom=0, left=0, right=0, color=255):
    return np.pad(img, [(top, bottom), (left, right), (0, 0)], "reflect")


def hiseq_color_cv2_img(img):
    (b, g, r) = cv2.split(img)
    bH        = cv2.equalizeHist(b)
    gH        = cv2.equalizeHist(g)
    rH        = cv2.equalizeHist(r)
    result    = cv2.merge((bH, gH, rH))
    return result


def auto_padding(img, times=16):
    # img: numpy image with shape H*W*C
    h, w, _ = img.shape
    h1, w1  = (times - h % times) // 2, (times - w % times) // 2
    h2, w2  = (times - h % times) - h1, (times - w % times) - w1
    img     = cv2.copyMakeBorder(img, h1, h2, w1, w2, cv2.BORDER_REFLECT)
    return img, [h1, h2, w1, w2]


def format_measurements(meas):
    s_out = []
    for k, v in meas.items():
        v = f"{v:0.2f}" if isinstance(v, float) else v
        s_out.append(f"{k}: {v}")
    str_out = ", ".join(s_out)
    return str_out


# ----- Predict -----
@torch.no_grad()
def predict(args: dict | box.Box) -> str:
    cfg_path        = current_dir / "options" / args.cfg
    cfgs            = option.parse(str(cfg_path), is_train=False)
    cfgs["gpu_ids"] = None
    cfgs            = option.dict_to_nonedict(cfgs)
    
    # Start
    mon.print_run_summary(args)
    
    # Device
    device = mon.set_device(args.device)

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

    # Model
    model = create_model(cfgs)
    model.load_network(load_path=pretrained, network=model.netG)
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
            image, padding_params = auto_padding(image)
            his    = hiseq_color_cv2_img(image)
            if cfgs.get("histeq_as_input", False):
                image = his
            lr_t = t(image)
            if cfgs["datasets"]["train"].get("log_low", False):
                lr_t = torch.log(torch.clamp(lr_t + 1e-3, min=1e-3))
            if cfgs.get("concat_histeq", False):
                his  = t(his)
                lr_t = torch.cat([lr_t, his], dim=1)
            lr_t = lr_t.to(device)
            heat = cfgs["heat"]
            timers.preprocess.tock()

            # Infer
            timers.infer.tick()
            sr_t = model.get_sr(lq=lr_t, heat=None)
            timers.infer.tock()

            # Postprocess
            timers.postprocess.tick()
            enhanced = rgb(
                torch.clamp(sr_t, 0, 1)[
                    :, :,
                    padding_params[0]:sr_t.shape[2] - padding_params[1],
                    padding_params[2]:sr_t.shape[3] - padding_params[3]
                ]
            )
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
