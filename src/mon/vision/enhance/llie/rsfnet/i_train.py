#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements the paper: "Zero-Reference Deep Curve Estimation for Low-Light Image
Enhancement," CVPR 2020.

References:
    - https://github.com/sophont01/RSFNet
"""

import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.optim

import mon
from libs.src.model import RRNet

torch.autograd.set_detect_anomaly(True)
warnings.filterwarnings("ignore", category=FutureWarning)
eps = np.finfo(np.float32).eps

current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# ----- Train -----
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0.0, 0.02)


def train(args: dict) -> str:
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
    resize       = args["resize"]
    epochs       = args["epochs"]
    steps        = args["steps"]
    benchmark    = args["benchmark"]
    save_image   = args["save_image"]
    save_debug   = args["save_debug"]
    keep_subdirs = args["keep_subdirs"]
    verbose      = args["verbose"]
    
    factors      = args["network"]["factors"]
    freeze       = args["network"]["freeze"]
    f_over_exp   = args["network"]["f_over_exp"]
    lr           = args["optimizer"]["lr"]
    lr_decay     = args["optimizer"]["lr_decay"]
    max_norm     = args["optimizer"]["max_norm"]
    
    # Start
    mon.console.rule(f"[bold red] {fullname}")
    mon.console.log(f"Machine: {hostname}")
    
    # Device
    device = mon.set_device(device)

    # Seed
    mon.set_random_seed(seed)
    
    # Data I/O
    datamodule: mon.DataModule = mon.DATAMODULES.build(config=args["datamodule"])
    datamodule.setup(stage="train")
    train_dataloader = datamodule.train_dataloader
    
    # Model
    model = RRNet(mode="train", **args["network"]).to(device)
    if weights is not None and mon.Path(weights).is_weights_file():
        model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    model.apply(weights_init)
    
    # Optimizer
    optimizer = torch.optim.SGD([
        {"params": model.fuseNet.encoder.parameters(), "lr": lr},
        {"params": model.fuseNet.decoder.parameters(), "lr": lr},
    ])
    for i in range(factors):
        optimizer.add_param_group({"params": model.factNet.lmbda_A[i].parameters(), "lr": 0.01})  # 0.01
        optimizer.add_param_group({"params": model.factNet.lmbda_E[i].parameters(), "lr": 0.01})  # 0.01
        optimizer.add_param_group({"params": model.factNet.step[i].parameters(),    "lr": 0.01})  # 0.01
        
    # Training
    for epoch in range(epochs):
        model.train()
        model.factNet.et_mean = [[] for _ in range(factors)]
        model.L = dict.fromkeys(("L_color", "L_exp", "L_TV", "L_fact"))
        
        if epoch > freeze + 25:
            optimizer.param_groups[0]["lr"] = optimizer.param_groups[0]["lr"] * lr_decay
            optimizer.param_groups[1]["lr"] = optimizer.param_groups[1]["lr"] * lr_decay
            
        with mon.create_progress_bar() as pbar:
            for i, datapoint in pbar.track(
                sequence    = enumerate(train_dataloader),
                total       = len(train_dataloader),
                description = f"[bright_yellow] Training"
            ):
                image = datapoint["image"].to(device)

                enhanced, loss = model(image)
                if f_over_exp:
                    enhanced = 1 - enhanced
                model.freeze_fact(epoch)
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)  # LOL-v1, LOL-v2-real, LOL-v2-synthetic
                optimizer.step()
                
                del enhanced, loss
        
        # Log
        for i in range(factors):
            print(f"\t E[{i}][0]={model.factNet.lmbda_E[i][0].item():0.9f} "
                  f"\t A[{i}][0]={model.factNet.lmbda_A[i][0].item():0.9f} "
                  f"\t step[{i}][0]={model.factNet.step[i][0].item():0.9f} ")
            
        # Save
        torch.save(model.state_dict(), save_dir / f"{fullname}_last.pt")


# ----- Main -----
def main() -> str:
    args = mon.parse_train_args(model_root=current_dir)
    train(args)


if __name__ == "__main__":
    main()
