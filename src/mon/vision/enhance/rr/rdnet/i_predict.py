#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements the paper: "Reversible Decoupling Network for Single Image Reflection
Removal," CVPR 2025.

References:
    - https://github.com/lime-j/RDNet
"""

import box
import model as mmodel
import torch.optim

import mon
from models import make_model
from options.net_options.train_options import TrainOptions

current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# ----- Utils -----
def benchmark(model: torch.nn.Module):
    flops, params = mon.compute_efficiency_score(model=model)
    mon.console.log(f"FLOPs : {flops:.4f}")
    mon.console.log(f"Params: {params:.4f}")


def tensor2im(image_tensor):
    image_tensor = image_tensor.detach()
    image_numpy  = image_tensor[0].cpu().float().numpy()
    image_numpy  = np.clip(image_numpy, 0, 1)
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy  = (np.transpose(image_numpy, (1, 2, 0))) * 255.0
    return image_numpy


# ----- Predict -----
@torch.no_grad()
def predict(args: dict | box.Box) -> str:
    opt = TrainOptions().parse()
    opt.isTrain    = False
    opt.no_log     = True
    opt.display_id = 0
    opt.verbose    = False

    # Start
    mon.print_run_summary(args)

    # Device
    device          = mon.set_device(args.device)
    cudnn.benchmark = True
    opt.device      = device
    
    # Seed
    mon.set_random_seed(args.seed)

    # Data I/O
    data_name, data_loader = mon.parse_data_loader(args.data, args.root, True, verbose=False)

    # Model
    opt.net_c_path = str(args.weights / "cls_model.pth")
    opt.icnn_path  = str(args.weights / "rdnet.pth")
    model = make_model(opt.model)
    model.initialize(opt)
    model.net_i.eval()
    model.net_c.eval()
    
    # Benchmark
    if args.benchmark:
        benchmark(model)
    
    # Predict
    timers = mon.TimeProfiler()
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
            if args.resize and h0 != args.imgsz[0] and w0 != args.imgsz[1]:
                image = mon.resize(image, size=args.imgsz)
            image  = image.to(device)
            timers.preprocess.tock()

            # Infer
            timers.infer.tick()
            model.set_input(
                data={
                    "input": image,
                    "fn"   : path,
                },
                mode="test"
            )
            outputs = model.forward()
            timers.infer.tock()

            # Postprocess
            timers.postprocess.tick()
            output_i, output_j = outputs
            if args.resize and h0 != args.imgsz[0] and w0 != args.imgsz[1]:
                output_i = mon.resize(output_i, size=(h0, w0))
                output_j = mon.resize(output_j, size=(h0, w0))
            output_i = tensor2im(output_i)
            output_j = tensor2im(output_j)
            timers.postprocess.tock()
            
            # Save
            if args.save_image:
                out_dir  = mon.parse_output_dir(args.save_dir, data_name, mon.SAVE_IMAGE_DIR, path, args.keep_subdirs, args.save_nearby)
                out_path = out_dir / f"{path.stem}{mon.SAVE_IMAGE_EXT}"
                mon.save_image(output_i, out_path)

            if args.save_debug:
                out_dir  = mon.parse_output_dir(args.save_dir, data_name, mon.SAVE_DEBUG_DIR, path, args.keep_subdirs, args.save_nearby)
                out_path = out_dir / f"{path.stem}{mon.SAVE_IMAGE_EXT}"
                mon.save_image(output_j, out_path)

    # Finish
    timers.print()
    return str(args.save_dir)


# ----- Main -----
def main() -> str:
    args = mon.parse_predict_args(model_root=current_dir)
    predict(args)


if __name__ == "__main__":
    main()
