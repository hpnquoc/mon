#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements the paper: "Towards Efficient and Scale-Robust Ultra-High-Definition
Image Demoireing," ECCV 2022.

References:
    https://github.com/CVMI-Lab/UHDM
"""

import box

import mon
from model.nets import my_model
from utils.common import *
from utils.loss_util import *

current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# ----- Utils -----
def benchmark(model: torch.nn.Module):
    flops, params = mon.compute_efficiency_score(model=model)
    mon.console.log(f"FLOPs : {flops:.4f}")
    mon.console.log(f"Params: {params:.4f}")


# ----- Predict -----
@torch.no_grad()
def predict(args: dict | box.Box) -> str:
    # Start
    mon.print_run_summary(args)
    
    # Device
    device = mon.set_device(args.device)
    
    # Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if args.seed == 0:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark     = True
    
    # Data I/O
    data_name, data_loader = mon.parse_data_loader(args.data, args.root, True, verbose=False)
    
    # Pretrained
    pretrained = args.resume
    if args.weights and args.weights.is_weights_file(exist=True):
        pretrained = args.weights
    if pretrained and pretrained.is_weights_file():
        mon.console.log(f"Pretrained: {pretrained}.")
        model_state_dict = torch.load(pretrained)["state_dict"]
        if  pretrained.is_weights_file():
            model_state_dict = torch.load(pretrained, weights_only=True)
    else:
        raise ValueError(f"Invalid weights file: {pretrained}.")

    # Model
    model = my_model(
        en_feature_num = args.MODEL.EN_FEATURE_NUM,
        en_inter_num   = args.MODEL.EN_INTER_NUM,
        de_feature_num = args.MODEL.DE_FEATURE_NUM,
        de_inter_num   = args.MODEL.DE_INTER_NUM,
        sam_number     = args.MODEL.SAM_NUMBER,
    )
    model.load_state_dict(model_state_dict)
    model = model.to(device)
    model.eval()
    
    # Benchmark
    if args.benchmark:
        benchmark(model)
        
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
            # if args.resize and h0 != args.imgsz[0] and w0 != args.imgsz[1]:
            #     image = mon.resize(image, size=args.imgsz)
            # if h0 != 2000 or w0 != 2992:
            #     image = mon.resize(image, [2000, 2992])
            # Pad image such that the resolution is a multiple of 32
            b, c, h, w = image.size()
            w_pad      = (math.ceil(w / 32) * 32 - w) // 2
            h_pad      = (math.ceil(h / 32) * 32 - h) // 2
            w_odd_pad  = w_pad
            h_odd_pad  = h_pad
            if w % 2 == 1:
                w_odd_pad += 1
            if h % 2 == 1:
                h_odd_pad += 1
            image = img_pad(image, w_pad=w_pad, h_pad=h_pad, w_odd_pad=w_odd_pad, h_odd_pad=h_odd_pad)
            image = image.to(device)
            timers.preprocess.tock()

            # Infer
            timers.infer.tick()
            outputs = model(image)
            timers.infer.tock()

            # Postprocess
            timers.postprocess.tick()
            out_1, _, _ = outputs
            if h_pad != 0:
                out_1 = out_1[:, :, h_pad:-h_odd_pad, :]
            if w_pad != 0:
                out_1 = out_1[:, :, :, w_pad:-w_odd_pad]
            enhanced = out_1.detach().cpu()
            # if h0 != 2000 or w0 != 2992:
            #     enhanced = mon.resize(enhanced, [h0, w0])
            timers.postprocess.tock()

            # Save
            if args.save_image:
                out_dir  = mon.parse_output_dir(args.save_dir, data_name, mon.SAVE_IMAGE_DIR, path, args.keep_subdirs, args.save_nearby)
                out_path = out_dir / f"{path.stem}{mon.SAVE_IMAGE_EXT}"
                mon.save_image(enhanced, out_path)
    
    # Finish
    timers.print()
    return str(args.save_dir)
    

# ----- Main -----
def main() -> str:
    args = mon.parse_predict_args(model_root=current_dir)
    predict(args)


if __name__ == "__main__":
    main()
