#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""PSENet model prediction pipeline for low-light image enhancement.

References:
    - Paper: "PSENet: Progressive Self-Enhancement Network for Unsupervised
      Extreme-Light Image Enhancement," WACV 2023.
    - Code: https://github.com/VinAIResearch/PSENet-Image-Enhancement
"""

import box
import torch
import torch.optim

import mon
from mon.vision.enhance.lle import psenet

current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# ----- Utils -----
def benchmark(model: torch.nn.Module):
    flops, params = mon.compute_efficiency_score(model=model)
    mon.console.log(f"Params    : {params:.4f}")
    mon.console.log(f"FLOPs     : {flops:.4f}")


def read_pytorch_lightning_state_dict(ckpt):
    new_state_dict = {}
    for k, v in ckpt["state_dict"].items():
        if k.startswith("model."):
            new_state_dict[k[len("model.") :]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


# ----- Predict -----
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
        state_dict = read_pytorch_lightning_state_dict(torch.load(str(pretrained), weights_only=False))
    else:
        raise ValueError(f"Invalid weights file: {pretrained}.")

    # Model
    model = psenet.PSENet()
    model.load_state_dict(state_dict)
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
            enhanced, _ = outputs
            if args.resize and (h0 != args.imgsz[0] or w0 != args.imgsz[1]):
                enhanced = mon.resize(enhanced, size=(h0, w0))
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
