#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements the paper: "Fast Context-Based Low-Light Image Enhancement via Neural
Implicit Representations," ECCV 2024.

References:
    - https://github.com/ctom2/colie
"""

import box
import kornia
import thop
import torch.optim
from fvcore.nn import FlopCountAnalysis, parameter_count

import mon
from color import hsv2rgb_torch, rgb2hsv_torch
from loss import *
from mon.nn import _size_2_t
from mon.vision.enhance.lle.colie.siren import INF_FF_FINER
from siren import INF
from utils import *

current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# ----- Utils -----
def compute_efficiency_score(model: torch.nn.Module, image_size: _size_2_t = 512) -> tuple[float, float]:
    """Computes FLOPs and parameters for a model.

    Args:
        model: PyTorch model to profile.
        image_size: Input image size (H, W) or single int. Default is ``512``.

    Returns:
        Tuple of (FLOPs, parameters) as floats.
    """
    patches = torch.rand(image_size, image_size, 49).to(mon.get_model_device(model))
    coords  = torch.rand(image_size, image_size,  2).to(mon.get_model_device(model))
    
    flops, params = thop.profile(model, inputs=(patches, coords,), verbose=False)
    flops   = FlopCountAnalysis(model, input).total() if flops == 0 else flops
    params  = model.params           if hasattr(model, "params") and params == 0 else params
    params  = parameter_count(model) if hasattr(model, "params") else params
    params  = sum(params.values())   if isinstance(params, dict) else params

    return flops, params


def benchmark(model: torch.nn.Module):
    flops, params = compute_efficiency_score(model=model)
    mon.console.log(f"Params    : {params:.4f}")
    mon.console.log(f"FLOPs     : {flops:.4f}")


# ----- Predict -----
def predict(args: dict | box.Box) -> str:
    inr          = args.network.inr
    window       = args.network.window
    num_layers   = args.network.num_layers
    hidden_dim   = args.network.hidden_dim
    add_layer    = args.network.add_layer
    lr           = args.optimizer.lr
    weight_decay = args.optimizer.weight_decay
    L            = args.loss.L
    alpha        = args.loss.alpha
    beta         = args.loss.beta
    gamma        = args.loss.gamma
    delta        = args.loss.delta
    
    # Start
    mon.print_run_summary(args)

    # Device
    device = mon.set_device(args.device)

    # Seed
    mon.set_random_seed(args.seed)

    # Data I/O
    data_name, data_loader = mon.parse_data_loader(args.data, args.root, True, verbose=False)

    # Benchmark
    if args.benchmark:
        if inr == "ff_finer":
            model = INF_FF_FINER(patch_dim=window ** 2, num_layers=num_layers, hidden_dim=hidden_dim, add_layer=add_layer)
        else:
            model = INF(patch_dim=window ** 2, num_layers=num_layers, hidden_dim=hidden_dim, add_layer=add_layer)
        flops, params = compute_efficiency_score(model=model)
        mon.console.log(f"Params    : {params:.4f}")
        mon.console.log(f"FLOPs     : {flops:.4f}")

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
            meta     = datapoint["meta"]
            path     = mon.Path(datapoint["meta"]["path"])
            img_rgb  = get_image(str(path)).to(device)
            # img_hsv  = rgb2hsv_torch(img_rgb).to(device)
            img_hsv  = kornia.color.rgb_to_hsv(img_rgb).to(device)
            img_v    = get_v_component(img_hsv).to(device)
            img_v_lr = interpolate_image(img_v, args.imgsz[0], args.imgsz[1]).to(device)
            coords   = get_coords(args.imgsz[0], args.imgsz[1]).to(device)
            patches  = get_patches(img_v_lr, window).to(device)
            timers.preprocess.tock()

            # Model
            if inr == "ff_finer":
                model = INF_FF_FINER(patch_dim=window ** 2, num_layers=num_layers, hidden_dim=hidden_dim, add_layer=add_layer)
            else:
                model = INF(patch_dim=window ** 2, num_layers=num_layers, hidden_dim=hidden_dim, add_layer=add_layer)
            model = model.to(device)
            # Optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)
            # Loss
            l_exp = L_exp(16, L)
            l_TV  = L_TV()

            # Optimize
            timers.infer.tick()
            for epoch in range(args.epochs):
                model.train()
                optimizer.zero_grad()
                #
                illu_res_lr    = model(patches, coords)
                illu_res_lr    = illu_res_lr.view(1, 1,  args.imgsz[0], args.imgsz[1])
                illu_lr        = illu_res_lr + img_v_lr
                img_v_fixed_lr = img_v_lr / (illu_lr + 1e-4)
                #
                loss_spa       = torch.mean(torch.abs(torch.pow(illu_lr - img_v_lr, 2))) * alpha
                loss_tv        = l_TV(illu_lr) * beta
                loss_exp       = torch.mean(l_exp(illu_lr)) * gamma
                loss_sparsity  = torch.mean(img_v_fixed_lr) * delta
                loss           = loss_spa * alpha + loss_tv * beta + loss_exp * gamma + loss_sparsity * delta  # ???
                loss.backward()
                optimizer.step()
            timers.infer.tock()

            # Postprocess
            timers.postprocess.tick()
            img_v_fixed   = filter_up(img_v_lr, img_v_fixed_lr, img_v)
            img_hsv_fixed = replace_v_component(img_hsv, img_v_fixed)
            # img_rgb_fixed = hsv2rgb_torch(img_hsv_fixed)
            img_rgb_fixed = kornia.color.hsv_to_rgb(img_hsv_fixed)
            img_rgb_fixed = img_rgb_fixed / torch.max(img_rgb_fixed)
            enhanced      = (torch.movedim(img_rgb_fixed, 1, -1)[0].detach().cpu().numpy() * 255).astype(np.uint8)
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
