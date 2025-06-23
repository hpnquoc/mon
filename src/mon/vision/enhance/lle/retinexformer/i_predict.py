#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements the paper: "Retinexformer: One-stage Retinex-based Transformer for
Low-light Image Enhancement," ICCV 2023.

References:
    - https://github.com/caiyuanhao1998/Retinexformer
"""

import box
import torch
import torch.nn.functional as F

import mon
from basicsr.models import create_model
from basicsr.utils.options import parse

current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# ----- Utils -----
def benchmark(model: torch.nn.Module):
    flops, params = mon.compute_efficiency_score(model=model)
    mon.console.log(f"Params: {params:.4f}")
    mon.console.log(f"FLOPs : {flops:.4f}")


# ----- Predict -----
@torch.no_grad()
def predict(args: dict | box.Box) -> str:
    cfg_path = current_dir / "option" / args.cfg
    cfgs     = parse(str(cfg_path), is_train=False)
    
    # Start
    mon.print_run_summary(args)
    
    # Device
    # gpu_list = ",".join(str(x) for x in args.gpus)
    # os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
    # print("export CUDA_VISIBLE_DEVICES=" + gpu_list)
    device = mon.set_device(args.device)
    cfgs["dist"]   = False
    cfgs["device"] = device

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
        checkpoint = torch.load(pretrained)
    else:
        raise ValueError(f"Invalid weights file: {pretrained}.")

    # Model
    model = create_model(cfgs).net_g
    try:
        model.load_state_dict(checkpoint["params"])
    except:
        new_checkpoint = {}
        for k in checkpoint["params"]:
            new_checkpoint["module." + k] = checkpoint["params"][k]
        model.load_state_dict(new_checkpoint)
    model = model.to(device)
    model.eval()
    
    # Benchmark
    if args.benchmark:
        benchmark(model)

    # Predict
    factor = 4
    timers = mon.TimeProfiler()
    timers.total.tick()
    with mon.create_progress_bar() as pbar:
        for i, datapoint in pbar.track(
            sequence    = enumerate(data_loader),
            total       = len(data_loader),
            description = f"[bright_yellow]Predicting"
        ):
            if torch.cuda.is_available():
                torch.cuda.ipc_collect()
                torch.cuda.empty_cache()

            # Preprocess
            timers.preprocess.tick()
            path   = mon.Path(datapoint["meta"]["path"])
            image  = datapoint["image"]
            h0, w0 = mon.image_size(image)
            if args.resize and h0 != args.imgsz[0] and w0 != args.imgsz[1]:
                image = mon.resize(image, args.imgsz)
                # mon.console.log("Resizing images to: ", image.shape[2], image.shape[3])
                # images = proc.resize(input=images, size=[1000, 666])
            # Padding in case images are not multiples of 4
            h, w  = mon.image_size(image)
            H, W  = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
            padh  = H - h if h % factor != 0 else 0
            padw  = W - w if w % factor != 0 else 0
            image = F.pad(image, (0, padw, 0, padh), 'reflect')
            image = image.to(device)
            timers.preprocess.tock()

            # Infer
            timers.infer.tick()
            outputs = model(image)
            timers.infer.tock()
            
            # Postprocess
            timers.postprocess.tick()
            # Unpad images to original dimensions
            enhanced = outputs[:, :, :h, :w]
            if args.resize and h0 != args.imgsz[0] and w0 != args.imgsz[1]:
                enhanced = mon.resize(enhanced, (h0, w0))
            enhanced = torch.clamp(enhanced, 0, 1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
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
