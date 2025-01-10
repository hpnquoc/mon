#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse

import torch
import torch.optim

import mon
import json
import os
import random
import warnings
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from libs.full.datasets.datasets import MyDataset
from libs.full.src.v8.model import RRNet
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

torch.autograd.set_detect_anomaly(True)
warnings.filterwarnings("ignore", category=FutureWarning)
eps = np.finfo(np.float32).eps

console      = mon.console
current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# region Train

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0.0, 0.02)
        
        
def train(args: argparse.Namespace):
    # General config
    data     = args.data
    fullname = args.fullname
    save_dir = mon.Path(args.save_dir)
    weights  = args.weights
    device   = mon.set_device(args.device)
    epochs   = args.epochs
    verbose  = args.verbose
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Directory
    weights_dir = save_dir
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    # Model
    model = RRNet(args).to(device)
    if weights is not None and mon.Path(weights).is_weights_file():
        model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    model.apply(weights_init)
    
    # Optimizer
    optimizer = torch.optim.SGD([
        {"params": model.fuseNet.encoder.parameters(), "lr": args.lr},
        {"params": model.fuseNet.decoder.parameters(), "lr": args.lr},
    ])
    for i in range(args.factors):
        optimizer.add_param_group({"params": model.factNet.lmbda_A[i].parameters(), "lr": 0.01})  # 0.01
        optimizer.add_param_group({"params": model.factNet.lmbda_E[i].parameters(), "lr": 0.01})  # 0.01
        optimizer.add_param_group({"params": model.factNet.step[i].parameters(),    "lr": 0.01})  # 0.01
        
    # Data I/O
    train_dataset = dataloader.lowlight_loader(args.data)
    train_loader  = torch.utils.data.DataLoader(
        train_dataset,
        batch_size  = args.train_batch_size,
        shuffle     = True,
        num_workers = args.num_workers,
        pin_memory  = True
    )
    
    # Training
    for epoch in range(0, epochs):
        dic = {"train_loss": 0, "L_color": 0, "L_exp": 0, "L_TV": 0, "L_fact": 0}
        model.train()
        model.factNet.et_mean = [[] for i in range(args.factors)]
        model.L = dict.fromkeys(("L_color", "L_exp", "L_TV", "L_fact"))
        if epoch > args.freeze + 25:
            optimizer.param_groups[0]["lr"] = optimizer.param_groups[0]["lr"] * args.lr_decay
            optimizer.param_groups[1]["lr"] = optimizer.param_groups[1]["lr"] * args.lr_decay
    
        with mon.get_progress_bar() as pbar:
            for _, data in pbar.track(
                sequence    = enumerate(train_loader),
                total       = len(train_loader),
                description = f"[bright_yellow] Training"
            ):
                
                
                for iteration, img_lowlight in enumerate(train_loader):
                    img_lowlight = img_lowlight.to(device)
                    enhanced_image_1, enhanced_image, A = DCE_net(img_lowlight)
                    
                    loss_tv  = 200 * L_tv(A)
                    loss_spa = torch.mean(L_spa(enhanced_image, img_lowlight))
                    loss_col = 5   * torch.mean(L_color(enhanced_image))
                    loss_exp = 10  * torch.mean(L_exp(enhanced_image))
                    loss     = loss_tv + loss_spa + loss_col + loss_exp
        
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm(DCE_net.parameters(), args.grad_clip_norm)
                    optimizer.step()
                    
                    if ((iteration + 1) % args.display_iter) == 0:
                        print("Loss at iteration", iteration + 1, ":", loss.item())
                    if ((iteration + 1) % args.checkpoints_iter) == 0:
                        torch.save(DCE_net.state_dict(), weights_dir / "best.pt")

# endregion


# region Main

def main() -> str:
    args = mon.parse_train_args(model_root=current_dir)
    train(args)


if __name__ == "__main__":
    main()

# endregion
