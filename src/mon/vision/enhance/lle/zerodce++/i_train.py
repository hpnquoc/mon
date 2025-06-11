#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements the paper: "Learning to Enhance Low-Light Image via Zero-Reference Deep
Curve Estimation," IEEE TPAMI 2022.

References:
    - https://github.com/Li-Chongyi/Zero-DCE_extension
"""

import torch
import torch.optim
import box
import model as mmodel
import mon
import myloss

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
    mon.print_run_summary(args)

    # Device
    device = mon.set_device(args.device)

    # Seed
    mon.set_random_seed(args.seed)

    # Data I/O
    args["datamodule"] |= {
        "root"   : mon.parse_data_dir(args.root, args.datamodule.get("root", "")),
        "devices": device,
    }
    datamodule: mon.DataModule = mon.DATAMODULES.build(config=args.datamodule)
    datamodule.prepare_data()
    datamodule.setup(stage="train")
    train_dataloader = datamodule.train_dataloader

    # Pretrained
    pretrained = args.tuning
    if args.resume and args.resume.is_weights_file(exist=True):
        pretrained = args.resume
    if args.weights and args.weights.is_weights_file(exist=True):
        pretrained = args.weights
    if pretrained and pretrained.is_weights_file(exist=True):
        mon.console.log(f"Pretrained: {pretrained}.")
    else:
        mon.console.log(f"Pretrained: {None}, training from scratch.")

    # Model
    model = mmodel.enhance_net_nopool(args.network.scale_factor)
    model.apply(weights_init)
    if pretrained and pretrained.is_weights_file(exist=True):
        model.load_state_dict(torch.load(pretrained, weights_only=True))
    model = model.to(device)
    model.train()
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), **args.optimizer)
    
    # Loss
    L_color = myloss.L_color()
    L_spa   = myloss.L_spa()
    L_exp   = myloss.L_exp(16)
    L_tv    = myloss.L_TV()
    
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
                enhanced, r = model(image)
                
                # loss_tv = 200 * L_tv(A)
                loss_tv  = 1600 * L_tv(r)
                loss_spa = torch.mean(L_spa(enhanced, image))
                loss_col =    5 * torch.mean(L_color(enhanced))
                loss_exp =   10 * torch.mean(L_exp(enhanced, 0.6))
                loss     = loss_tv + loss_spa + loss_col + loss_exp
    
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()
                
                # Log
                if ((i + 1) % display_iter) == 0:
                    print("Loss at iteration", i + 1, ":", loss.item())
                
                # Save
                if ((i + 1) % checkpoints_iter) == 0:
                    torch.save(model.state_dict(), args.save_dir / "best.pt")


# ----- Main -----
def main() -> str:
    args = mon.parse_train_args(model_root=current_dir)
    train(args)


if __name__ == "__main__":
    main()
