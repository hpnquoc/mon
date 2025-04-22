#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements the paper: "Zero-Reference Deep Curve Estimation for Low-Light Image
Enhancement," CVPR 2020.

References:
    - https://github.com/Li-Chongyi/Zero-DCE
"""

import argparse
import random
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.optim

import mon
from libs.src.model import RRNet
from mon import albumentation as A

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
    
    factors  = args["network"]["factors"]
    lr       = args["optimizer"]["lr"]
    lr_step  = args["optimizer"]["lr_step"]
    lr_decay = args["optimizer"]["lr_decay"]
    
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
    with mon.create_progress_bar() as pbar:
        for _ in pbar.track(
            sequence    = range(epochs),
            total       = epochs,
            description = f"[bright_yellow] Training"
        ):
            for i, datapoint in enumerate(train_dataloader):
                image          = datapoint["image"].to(device)
                _, enhanced, r = dce_net(image)
                
                loss_tv  = 200 * L_tv(r)
                loss_spa = torch.mean(L_spa(enhanced, image))
                loss_col =   5 * torch.mean(L_color(enhanced))
                loss_exp =  10 * torch.mean(L_exp(enhanced))
                loss     = loss_tv + loss_spa + loss_col + loss_exp
    
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(dce_net.parameters(), grad_clip_norm)
                optimizer.step()
                
                # Log
                if ((i + 1) % display_iter) == 0:
                    print("Loss at iteration", i + 1, ":", loss.item())
                
                # Save
                if ((i + 1) % checkpoints_iter) == 0:
                    torch.save(dce_net.state_dict(), save_dir / "best.pt")


# ----- Main -----
def main() -> str:
    args = mon.parse_train_args(model_root=current_dir)
    train(args)


if __name__ == "__main__":
    main()
