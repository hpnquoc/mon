#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements ZINF model prediction pipeline for low-light image enhancement.

References:
    - Paper: "Zero-Shot Implicit Neural Fusion Network for Multimodal Low-Light
      Image Enhancement," arXiv 2025.
    - Code: https://github.com/phlong3105/mon
"""

import box
import thop
import torch
from fvcore.nn import FlopCountAnalysis, parameter_count

import mon
from mon import albumentations as A
from mon.vision.enhance.lle import zinf

current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# ----- Utils -----
def compute_complexity(model: mon.nn.Module, imgsz: int = 512) -> tuple[float, float]:
    """Computes FLOPs and parameters for a model.

    Args:
        model: PyTorch model to profile.
        imgsz: Input image size. Default: ``512``.

    Returns:
        A tuple of :math:`(flops, params)`.
    """
    # from fvcore.nn import parameter_count
    image      = torch.rand(1, 1, imgsz, imgsz)
    image_lr   = model.interpolate_image(image, imgsz)
    #
    spatial    = model.create_coords(imgsz)
    patch      = model.create_patches(image_lr, model.window_size)
    #
    spatial_ff = model.ff_embedding(spatial, model.B1)
    patch_ff   = model.ff_embedding(patch,   model.B2)
    
    flops, params = thop.profile(model, inputs=(spatial_ff, patch_ff, patch_ff, patch_ff, ), verbose=False)
    flops         = FlopCountAnalysis(model, input).total() if flops == 0 else flops
    params        = model.params           if hasattr(model, "params") and params == 0 else params
    params        = parameter_count(model) if hasattr(model, "params") else params
    params        = sum(params.values())   if isinstance(params, dict) else params
    
    return flops, params


def benchmark(model: mon.nn.Module):
    flops, params = compute_complexity(model=model)
    mon.log(f"Params    : {params:.4f}")
    mon.log(f"FLOPs     : {flops:.4f}")


# ----- Predict -----
def predict(args: dict | box.Box) -> str:
    mapping_func    = args.network.mapping_func
    window_size     = args.network.window_size
    hidden_dim      = args.network.hidden_dim
    num_layers      = args.network.num_layers
    add_layers      = args.network.add_layers
    use_ff          = args.network.use_ff
    nonlinear       = args.network.nonlinear
    depth_threshold = args.network.depth_threshold
    edge_threshold  = args.network.edge_threshold
    use_denoise     = args.network.use_denoise
    L               = args.network.L
    iters           = args.epochs
    
    # Start
    mon.rt.print_run_summary(args)

    # Device
    device = mon.create_device(args.device)

    # Seed
    mon.set_random_seed(args.seed)
    
    # Model
    model = zinf.ZINF(
        mapping_func    = mapping_func,
        window_size     = window_size,
        hidden_dim      = hidden_dim,
        num_layers      = num_layers,
        add_layers      = add_layers,
        use_ff          = use_ff,
        nonlinear       = nonlinear,
        depth_threshold = depth_threshold,
        edge_threshold  = edge_threshold,
        use_denoise     = use_denoise,
        L               = L,
        iters           = iters,
    )
    model = model.to(device)
    
    # Benchmark
    if args.benchmark:
        benchmark(model)
    
    # Data I/O
    transform = A.Compose([
        A.Normalize(normalization="min_max"),
        A.ToTensorV2(transpose_mask=True),
    ])
    data_name, dataloader = mon.data.build_dataloader(args.data, args.root, transform)
    
    # Predict
    timers = mon.TimeProfiler()
    timers.total.tick()
    with mon.create_progress_bar() as pbar:
        for i, datapoint in pbar.track(
            sequence    = enumerate(dataloader),
            total       = len(dataloader),
            description = f"[bright_yellow]Predicting"
        ):
            # Preprocess
            timers.preprocess.tick()
            meta  = datapoint["meta"][0]
            path  = mon.Path(meta["path"])
            image = datapoint["image"]
            depth = datapoint.get("depth", None)
            image = image.to(device)
            depth = depth.to(device) if depth is not None else None
            timers.preprocess.tock()

            # Optimize
            timers.infer.tick()
            outputs = model(image, depth)
            timers.infer.tock()

            # Postprocess
            timers.postprocess.tick()
            enhanced = outputs
            timers.postprocess.tock()
            
            # Save
            if args.save_image:
                out_dir  = mon.rt.parse_output_dir(args.save_dir, data_name, mon.SAVE_IMAGE_DIR, path, args.keep_subdirs, args.save_nearby)
                out_path = out_dir / f"{path.stem}{mon.SAVE_IMAGE_EXT}"
                mon.image.save_image(enhanced, out_path)
    timers.total.tock()

    # Finish
    timers.print()
    return str(args.save_dir)


# ----- Main -----
def main() -> str:
    args = mon.rt.parse_predict_args(model_root=current_dir)
    predict(args)


if __name__ == "__main__":
    main()
