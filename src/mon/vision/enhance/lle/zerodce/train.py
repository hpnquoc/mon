#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements Zero-DCE model training pipeline for low-light image enhancement.

References:
    - Paper: "Zero-Reference Deep Curve Estimation for Low-Light Image
      Enhancement," CVPR 2020.
    - Code: https://github.com/Li-Chongyi/Zero-DCE
"""

import box
import torch

import mon
from mon.vision.enhance.lle import zerodce
from . import loss as L

mon.dev()

current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# ----- Train -----
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def train(args: dict | box.Box) -> str:
    # Start
    mon.rt.print_run_summary(args)
    
    # Device
    device = mon.create_device(args.device)
    
    # Seed
    mon.set_random_seed(args.seed)
    
    # Pretrained
    pretrained = args.tuning
    if args.resume and args.resume.is_weights_file(exist=True):
        pretrained = args.resume
    if args.weights and args.weights.is_weights_file(exist=True):
        pretrained = args.weights
    if pretrained and pretrained.is_weights_file(exist=True):
        mon.log(f"Pretrained: {pretrained}.")
    else:
        mon.log(f"Pretrained: {None}, training from scratch.")

    # Model
    model = zerodce.ZeroDCE()
    model.apply(weights_init)
    if pretrained and pretrained.is_weights_file(exist=True):
        model.load_state_dict(torch.load(pretrained, weights_only=True))
    model = model.to(device)
    model.train()
    
    # Optimizer
    optimizer = mon.optim.Adam(model.parameters(), **args.optimizer)
    
    # Loss
    L_tv  = L.L_tv().to(device)
    L_spa = L.L_spa().to(device)
    L_col = L.L_col().to(device)
    L_exp = L.L_exp(16, 0.6).to(device)
    
    # Data I/O
    args["train_dataloader"]["datasets"]["root"] = mon.data.parse_data_dir(args.root)
    args["val_dataloader"]["datasets"]["root"]   = mon.data.parse_data_dir(args.root)
    train_dataloader = mon.data.DataLoader(**args.train_dataloader)
    val_dataloader   = mon.data.DataLoader(**args.val_dataloader)

    # Train
    grad_clip_norm   = args["trainer"]["grad_clip_norm"]
    display_iter     = args["trainer"]["display_iter"]
    checkpoints_iter = args["trainer"]["checkpoints_iter"]
    with mon.create_progress_bar() as pbar:
        for _ in pbar.track(
            sequence    = range(args.epochs),
            total       = args.epochs,
            description = f"[bright_yellow]Training"
        ):
            for i, datapoint in enumerate(train_dataloader):
                image       = datapoint["image"].to(device)
                outputs     = model(image)
                enhanced, r = outputs
                
                l_tv  = 200 * L_tv(r)
                l_spa =   1 * torch.mean(L_spa(enhanced, image))
                l_col =   5 * torch.mean(L_col(enhanced))
                l_exp =  10 * torch.mean(L_exp(enhanced))
                loss  = l_tv + l_spa + l_col + l_exp
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()
                
                # Log
                if args.verbose and ((i + 1) % display_iter) == 0:
                    mon.log(f"Iter: {i + 1} | Loss: {loss.item()}")
                
                # Save
                if ((i + 1) % checkpoints_iter) == 0:
                    torch.save(model.state_dict(), args.save_dir / "best.pt")


# ----- Main -----
def main() -> str:
    args = mon.rt.parse_train_args(model_root=current_dir)
    train(args)


if __name__ == "__main__":
    main()
