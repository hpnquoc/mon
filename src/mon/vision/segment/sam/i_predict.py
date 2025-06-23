#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements the paper: "Segment Anything," arXiv 2023.

References:
    https://github.com/facebookresearch/segment-anything
"""

import box
import numpy as np
import torch.optim

import mon
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

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
    model = sam_model_registry[args.network.model_type](checkpoint=pretrained)
    model = model.to(device)
    model.eval()
    mask_generator = SamAutomaticMaskGenerator(
        model                          = model,
        points_per_side                = args.network.points_per_side,
        points_per_batch               = args.network.points_per_batch,
        pred_iou_thresh                = args.network.pred_iou_thresh,
        stability_score_thresh         = args.network.stability_score_thresh,
        stability_score_offset         = args.network.stability_score_offset,
        box_nms_thresh                 = args.network.box_nms_thresh,
        crop_n_layers                  = args.network.crop_n_layers,
        crop_nms_thresh                = args.network.crop_nms_thresh,
        crop_n_points_downscale_factor = args.network.crop_n_points_downscale_factor,
        min_mask_region_area           = args.network.min_mask_region_area,
        output_mode                    = args.network.output_mode,
    )
    
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
            if args.resize and h0 != args.imgsz[0] and w0 != args.imgsz[1]:
                image = mon.resize(image, size=args.imgsz)
            image  = image.to(device)
            timers.preprocess.tock()
            
            # Infer
            timers.infer.tick()
            masks = mask_generator.generate(image)
            timers.infer.tock()

            # Postprocess
            timers.postprocess.tick()
            timers.postprocess.tock()

            # Save
            if args.save_image:
                out_dir = mon.parse_output_dir(args.save_dir, data_name, mon.SAVE_IMAGE_DIR, path, args.keep_subdirs, args.save_nearby)
                # Binary
                for j, mask in enumerate(masks):
                    out_path = out_dir / f"{path.stem}_mask_{j}{mon.SAVE_IMAGE_EXT}"
                    mon.save_image(np.uint8(mask["segmentation"]) * 255, out_path)

            if args.save_debug:
                debug_dir = mon.parse_output_dir(args.save_dir, data_name, mon.SAVE_DEBUG_DIR, path, args.keep_subdirs, args.save_nearby)
                if args.save_nearby:
                    debug_dir = debug_dir.parent / f"{debug_dir.stem}_c"
                # Color
                output = np.ones((masks[0]["segmentation"].shape[0], masks[0]["segmentation"].shape[1], 4))
                output[:, :, 3] = 0
                for _, mask in enumerate(masks):
                    mask_bool         = mask["segmentation"]
                    color_mask        = np.concatenate([np.random.random(3), [1.0]])  # 0.35
                    output[mask_bool] = color_mask
                debug_path = debug_dir / f"{path.stem}{mon.SAVE_IMAGE_EXT}"
                mon.save_image(np.uint8(output * 255), debug_path)
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
