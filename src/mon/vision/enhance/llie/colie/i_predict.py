#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements the paper: "Fast Context-Based Low-Light Image Enhancement via Neural
Implicit Representations," ECCV 2024.

References:
    - https://github.com/ctom2/colie
"""

from typing import Sequence

import torch.optim

import mon
from color import hsv2rgb_torch, rgb2hsv_torch
from loss import *
from siren import INF
from utils import *

current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# ----- Predict -----
def predict(args: dict) -> str:
    # Parse args
    hostname     = args["hostname"]
    root         = args["root"]
    data         = args["data"]
    fullname     = args["fullname"]
    save_dir     = args["save_dir"]
    weights      = args["weights"]
    device       = args["device"]
    seed         = args["seed"]
    imgsz        = args["imgsz"]
    imgsz        = imgsz[0] if isinstance(imgsz, Sequence) else imgsz
    resize       = args["resize"]
    epochs       = args["epochs"]
    steps        = args["steps"]
    benchmark    = args["benchmark"]
    save_image   = args["save_image"]
    save_debug   = args["save_debug"]
    use_fullpath = args["use_fullpath"]
    verbose      = args["verbose"]
    
    window       = args["network"]["window"]
    num_layers   = args["network"]["num_layers"]
    hidden_dim   = args["network"]["hidden_dim"]
    add_layer    = args["network"]["add_layer"]
    lr           = args["optimizer"]["lr"]
    weight_decay = args["optimizer"]["weight_decay"]
    L            = args["loss"]["L"]
    alpha        = args["loss"]["alpha"]
    beta         = args["loss"]["beta"]
    gamma        = args["loss"]["gamma"]
    delta        = args["loss"]["delta"]
    
    # Start
    mon.console.rule(f"[bold red] {fullname}")
    mon.console.log(f"Machine: {hostname}")
    
    # Device
    device = mon.set_device(device)
    
    # Seed
    mon.set_random_seed(seed)
    
    # Data I/O
    mon.console.log(f"[bold red]{data}")
    data_name, data_loader = mon.parse_data_loader(data, root, True, verbose=False)
    
    # Benchmark
    if benchmark:
        model = INF(patch_dim=window ** 2, num_layers=num_layers, hidden_dim=hidden_dim, add_layer=add_layer)
        flops, params = mon.compute_efficiency_score(model=model, image_size=512)
        mon.console.log(f"FLOPs : {flops:.4f}")
        mon.console.log(f"Params: {params:.4f}")
    
    # Predicting
    timer = mon.Timer()
    with mon.create_progress_bar() as pbar:
        for i, datapoint in pbar.track(
            sequence    = enumerate(data_loader),
            total       = len(data_loader),
            description = f"[bright_yellow] Predicting"
        ):
            # Input
            meta       = datapoint["meta"]
            image_path = mon.Path(meta["path"])
            img_rgb    = get_image(str(image_path))
            img_hsv    = rgb2hsv_torch(img_rgb)
            img_v      = get_v_component(img_hsv)
            img_v_lr   = interpolate_image(img_v, imgsz, imgsz)
            coords     = get_coords(imgsz, imgsz)
            patches    = get_patches(img_v_lr, window)
            
            # Model
            img_siren  = INF(patch_dim=window ** 2, num_layers=num_layers, hidden_dim=hidden_dim, add_layer=add_layer)
            img_siren  = img_siren.to(device)
            # Optimizer
            optimizer  = torch.optim.Adam(img_siren.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)
            # Loss
            l_exp = L_exp(16, L)
            l_TV  = L_TV()

            # Optimize
            timer.tick()
            for epoch in range(epochs):
                img_siren.train()
                optimizer.zero_grad()
                #
                illu_res_lr    = img_siren(patches, coords)
                illu_res_lr    = illu_res_lr.view(1, 1, imgsz, imgsz)
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
            img_v_fixed   = filter_up(img_v_lr, img_v_fixed_lr, img_v)
            img_hsv_fixed = replace_v_component(img_hsv, img_v_fixed)
            img_rgb_fixed = hsv2rgb_torch(img_hsv_fixed)
            img_rgb_fixed = img_rgb_fixed / torch.max(img_rgb_fixed)
            timer.tock()
            
            # Save
            if save_image:
                if use_fullpath:
                    rel_path    = image_path.relative_path(data_name)
                    output_path = save_dir / rel_path.parent / f"{image_path.stem}.jpg"
                else:
                    output_path = save_dir / data_name / f"{image_path.stem}.jpg"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray((torch.movedim(img_rgb_fixed, 1, -1)[0].detach().cpu().numpy() * 255).astype(np.uint8)).save(str(output_path))
    
    # Finish
    mon.console.log(f"Average time: {timer.avg_time}")


# ----- Main -----
def main() -> str:
    args = mon.parse_predict_args(model_root=current_dir)
    predict(args)


if __name__ == "__main__":
    main()
