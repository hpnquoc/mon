#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import random
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.optim

import mon
from libs.full.src.v8.model import RRNet
from mon import albumentation as A

torch.autograd.set_detect_anomaly(True)
warnings.filterwarnings("ignore", category=FutureWarning)
eps = np.finfo(np.float32).eps

console      = mon.console
current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# ----- Train -----

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0.0, 0.02)
        
        
def train(args: argparse.Namespace):
    # General config
    fullname = args.fullname
    save_dir = mon.Path(args.save_dir)
    weights  = args.weights
    device   = mon.set_device(args.device)
    imgsz    = args.imgsz
    epochs   = args.epochs
    verbose  = args.verbose
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Directory
    weights_dir = save_dir
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    # Model
    args.mode   = "train"
    args.device = device
    model = RRNet(args)
    if weights is not None and mon.Path(weights).is_weights_file():
        model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    model = model.to(device)
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
    data_args = {
        "name"      : args.data,
        "root"      : mon.DATA_DIR / "enhance",
        "transform" : A.Compose(transforms=[
            A.Resize(width=imgsz, height=imgsz),
        ]),
        "to_tensor" : True,
        "cache_data": False,
        "batch_size": args.batch_size,
        "devices"   : device,
        "shuffle"   : True,
        "verbose"   : verbose,
    }
    datamodule: mon.DataModule = mon.DATAMODULES.build(config=data_args)
    datamodule.prepare_data()
    datamodule.setup(stage="train")
    train_dataloader = datamodule.train_dataloader
    val_dataloader   = datamodule.val_dataloader
    
    # Training
    for epoch in range(0, epochs):
        dic = {"train_loss": 0, "L_color": 0, "L_exp": 0, "L_TV": 0, "L_fact": 0}
        model.train()
        model.factNet.et_mean = [[] for i in range(args.factors)]
        model.L = dict.fromkeys(("L_color", "L_exp", "L_TV", "L_fact"))
        if epoch > args.freeze + 25:
            optimizer.param_groups[0]["lr"] = optimizer.param_groups[0]["lr"] * args.lr_decay
            optimizer.param_groups[1]["lr"] = optimizer.param_groups[1]["lr"] * args.lr_decay
    
        with mon.create_progress_bar() as pbar:
            for i, datapoint in pbar.track(
                sequence    = enumerate(train_dataloader),
                total       = len(train_dataloader),
                description = f"[bright_yellow] Training"
            ):
                optimizer.zero_grad()
                image      = datapoint["image"].to(device).type(torch.float32)
                ref        = datapoint["ref_image"].to(device).type(torch.float32)
                pred, loss = model(image, epoch)
                if args.f_OverExp:
                    pred = 1 - pred
                
                dic["train_loss"] += (loss.item()        / len(train_dataloader))
                dic["L_color"]    += (model.L["L_color"] / len(train_dataloader))
                dic["L_exp"]      += (model.L["L_exp"]   / len(train_dataloader))
                dic["L_TV"]       += (model.L["L_TV"]    / len(train_dataloader))
                dic["L_fact"]     += (model.L["L_fact"]  / len(train_dataloader))
                
                model.freezeFact(epoch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # for LOLv1, LOLv2, LOLsyn
                optimizer.step()
                del loss, pred
            
            """
            for j in range(args.factors):
                print(
                    f'''
                    \tE[{j}][0]={model.factNet.lmbda_E[j][0].item():0.9f}
                    \tA[{j}][0]={model.factNet.lmbda_A[j][0].item():0.9f}
                    \tstep[{j}][0]={model.factNet.step[j][0].item():0.9f}
                    '''
                )
            """
            torch.save(model.state_dict(), weights_dir / f"{fullname}_last.pt")
            



# ----- Main -----

def main() -> str:
    args = mon.parse_train_args(model_root=current_dir)
    train(args)


if __name__ == "__main__":
    main()
