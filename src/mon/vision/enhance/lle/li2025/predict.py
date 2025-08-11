#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Li2025 model prediction pipeline for low-light image enhancement.

References:
    - Paper: "Interpretable Unsupervised Joint Denoising and Enhancement for
      Real-World low-light Scenarios," ICLR 2025.
    - Code: https://github.com/huaqlili/unsupervised-light-enhance-ICLR2025
"""

import box
import torch.optim

import mon
from mon.vision.enhance.lle import li2025

current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# ----- Utils -----
def benchmark(model: torch.nn.Module):
    flops, params = mon.compute_efficiency_score(model=model)
    mon.console.log(f"Params    : {params:.4f}")
    mon.console.log(f"FLOPs     : {flops:.4f}")


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
    else:
        raise ValueError(f"Invalid weights file: {pretrained}.")

    # Model
    model = li2025.Li2025()
    model.load_state_dict(torch.load(pretrained, weights_only=True))
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
            L, _, R, X, I = outputs
            D = (image - X).cpu()
            I = torch.clamp(I, 0, 1).cpu()
            R = torch.clamp(R, 0, 1).cpu()
            L = torch.clamp(L, 0, 1).cpu()
            # L_img = transforms.ToPILImage()(L.squeeze(0))
            # R_img = transforms.ToPILImage()(R.squeeze(0))
            # I_img = transforms.ToPILImage()(I.squeeze(0))
            # D_img = transforms.ToPILImage()(D.squeeze(0))
            if args.resize and (h0 != args.imgsz[0] or w0 != args.imgsz[1]):
                L = mon.resize(L, size=(h0, w0))
                R = mon.resize(R, size=(h0, w0))
                I = mon.resize(I, size=(h0, w0))
                D = mon.resize(D, size=(h0, w0))
            timers.postprocess.tock()

            # Save
            if args.save_image:
                out_dir  = mon.parse_output_dir(args.save_dir, data_name, mon.SAVE_IMAGE_DIR, path, args.keep_subdirs, args.save_nearby)
                out_path = out_dir / f"{path.stem}{mon.SAVE_IMAGE_EXT}"
                mon.save_image(I, out_path)

            if args.save_debug:
                out_dir  = mon.parse_output_dir(args.save_dir, data_name, mon.SAVE_DEBUG_DIR, path, args.keep_subdirs, args.save_nearby)
                mon.save_image(L, out_dir / f"{path.stem}_L{mon.SAVE_IMAGE_EXT}")
                mon.save_image(R, out_dir / f"{path.stem}_R{mon.SAVE_IMAGE_EXT}")
                mon.save_image(D, out_dir / f"{path.stem}_D{mon.SAVE_IMAGE_EXT}")
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
