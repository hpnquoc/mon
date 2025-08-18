#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements ... model training pipeline for ...

References:
    - Paper: " ," arXiv 2025.
    - Code:
"""

import box
import torch
import torch.nn as nn

import mon
from mon import Path

mon.dev()

current_file = Path(__file__).absolute()
current_dir  = current_file.parents[0]


# ----- Train -----
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
    model: nn.Module = None
    if pretrained and pretrained.is_weights_file(exist=True):
        model.load_state_dict(torch.load(pretrained, weights_only=True))
    model = model.to(device)
    model.train()

    # Trainer
    # Optimizer
    # Scheduler
    # Loss

    # Data I/O
    args["train_dataloader"]["datasets"]["root"] = mon.data.parse_data_dir(args.root)
    args["val_dataloader"]["datasets"]["root"]   = mon.data.parse_data_dir(args.root)
    train_dataloader = mon.data.DataLoader(**args.train_dataloader)
    val_dataloader   = mon.data.DataLoader(**args.val_dataloader)
    
    # Train
    with mon.create_progress_bar() as pbar:
        for _ in pbar.track(
            sequence    = range(args.epochs),
            total       = args.epochs,
            description = f"[bright_yellow]Training"
        ):
            pass

    # Finish
    return str(args.save_dir)


# ----- Main -----
def main() -> str:
    args = mon.rt.parse_train_args(model_root=current_dir, verbose=False)
    train(args)


if __name__ == "__main__":
    main()
