#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements the paper: "DEIM: DETR with Improved Matching for Fast
Convergence," CVPR 2025.

References:
    - https://github.com/ShihuaHuang95/DEIM
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import box
import torch

import mon
from engine.core import YAMLConfig

current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# ----- Utils -----
def benchmark(model: torch.nn.Module):
    flops, params = mon.compute_efficiency_score(model=model)
    mon.console.log(f"FLOPs : {flops:.4f}")
    mon.console.log(f"Params: {params:.4f}")


# ----- Predict -----
class Model(torch.nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.model         = cfg.model.deploy()
        self.postprocessor = cfg.postprocessor.deploy()

    def forward(self, images, orig_target_sizes):
        outputs = self.model(images)
        outputs = self.postprocessor(outputs, orig_target_sizes)
        return outputs


@torch.no_grad()
def predict(args: dict | box.Box) -> str:
    # Start
    mon.print_run_summary(args)

    # Device
    device = mon.set_device(args.device)

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
    cfg_path     = current_dir / "option" / args.cfg
    updated_cfg  = args.updated_cfg
    updated_cfg |= {"resume": str(pretrained)} if pretrained else {}
    updated_cfg |= {
        "device": device,
        "seed"  : args.seed,
    }
    cfg = YAMLConfig(cfg_path=str(cfg_path), root=str(args.root), **updated_cfg)

    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    if pretrained:
        checkpoint = torch.load(pretrained, map_location="cpu")
        if "ema" in checkpoint:
            state = checkpoint["ema"]["module"]
        else:
            state = checkpoint["model"]
    else:
        raise AttributeError("Only support resume to load model.state_dict by now.")

    # Load train mode state and convert to deploy mode
    cfg.model.load_state_dict(state)
    model = Model(cfg).to(device)

    # Predict
    timers = mon.TimeProfiler()
    with mon.create_progress_bar() as pbar:
        for i, datapoint in pbar.track(
            sequence    = enumerate(data_loader),
            total       = len(data_loader),
            description = f"[bright_yellow] Predicting"
        ):
            # Preprocess
            timers.preprocess.tick()
            path   = mon.Path(datapoint["meta"]["path"])
            image  = datapoint["image"]
            h0, w0 = mon.image_size(image)
            size0  = torch.tensor([[w0, h0]]).to(device)
            if args.resize and h0 != args.imgsz[0] and w0 != args.imgsz[1]:
                image = mon.resize(image, size=args.imgsz)
            image  = image.to(device)
            timers.preprocess.tock()

            # Infer
            timers.infer.tick()
            outputs = model(image, size0)
            timers.infer.tock()

            # Postprocess
            timers.postprocess.tick()
            labels, boxes, scores = outputs
            labels = [l.cpu().numpy() for l in labels]
            boxes  = [b.cpu().numpy() for b in  boxes]
            scores = [s.cpu().numpy() for s in scores]
            timers.postprocess.tock()

            # Save Result
            if args.save_result:
                out_dir    = mon.parse_output_dir(args.save_dir, data_name, mon.SAVE_LABEL_DIR, path, args.keep_subdirs, args.save_nearby)
                label_path = out_dir / f"{path.stem}.txt"
                label_path.parent.mkdir(parents=True, exist_ok=True)
                with open(str(label_path), "w") as f:
                    for j, img in enumerate(image):
                        ss = scores[j]
                        cs = labels[j][ss >= args.conf_thres]
                        bs =  boxes[j][ss >= args.conf_thres]
                        if len(bs) == 0:
                            continue
                        bs = mon.convert_hbb(bbox=bs, fmt=mon.BBoxFormat.VOC2YOLO, height=h0, width=w0)
                        for c, b, s in zip(cs, bs, ss):
                            f.write(f"{c} {b[0]} {b[1]} {b[2]} {b[3]} {s}\n")

    # Finish
    timers.print()
    return str(args.save_dir)


# ----- Main -----
def main() -> str:
    args = mon.parse_predict_args(model_root=current_dir)
    predict(args)


if __name__ == "__main__":
    main()
