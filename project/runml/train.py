#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Train Pipeline.

This script trains a model on a given dataset.
"""

from __future__ import annotations

import argparse

import mon
import mon.core.utils

console      = mon.console
current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# region Train

def train(args: argparse.Namespace) -> str:
    # Parse args
    args         = vars(args)
    hostname     = args["hostname"]
    data         = args["data"]
    fullname     = args["fullname"]
    save_dir     = args["save_dir"]
    weights      = args["weights"]
    device       = args["device"]
    seed         = args["seed"]
    # imgsz        = args["imgsz"]
    # resize       = args["resize"]
    epochs       = args["epochs"]
    steps        = args["steps"]
    benchmark    = args["benchmark"]
    save_image   = args["save_image"]
    save_debug   = args["save_debug"]
    use_fullpath = args["use_fullpath"]
    verbose      = args["verbose"]
    
    # Start
    if mon.is_rank_zero():
        console.rule("[bold red] INITIALIZATION")
        console.log(f"Machine: {hostname}")
    
    # Device
    # device = mon.set_device(device)
    
    # Seed
    mon.set_random_seed(seed)
    
    # Data I/O
    datamodule: mon.DataModule = mon.DATAMODULES.build(config=args["mon"]["datamodule"])
    datamodule.prepare_data()
    datamodule.setup(stage="train")
    num_classes = getattr(datamodule.classlabels, "num_trainable_classes", None)
    num_classes = num_classes or args["mon"]["network"].get("num_classes", None)
    
    # Model
    args["mon"]["network"] |= {
        "num_classes": num_classes
    }
    args["mon"]["model"] |= {
        "fullname"   : fullname,
        "root"       : save_dir,
        "num_classes": num_classes,
        "weights"    : weights,
        "debug"      : save_debug,
        "verbose"    : verbose,
    }
    model: mon.Model = mon.MODELS.build(config=args["mon"]["model"])
    if mon.is_rank_zero():
        mon.print_dict(args, title=fullname)
        console.log("[green]Done")
    
    # Trainer
    if mon.is_rank_zero():
        console.rule("[bold red] SETUP TRAINER")
    
    callbacks = args["mon"]["trainer"]["callbacks"]
    for i, callback in enumerate(callbacks):
        if callback["name"] == "model_checkpoint":
            callbacks[i] |= {"filename": fullname}
    callbacks = mon.CALLBACKS.build_instances(configs=args["mon"]["trainer"]["callbacks"])
    ckpt      = mon.get_latest_checkpoint(dirpath=model.ckpt_dir)
    devices   = mon.to_int_list(device) if "auto" not in device else "auto"
    if args["mon"]["trainer"]["logger"]:
        logger = [mon.TensorBoardLogger(save_dir=save_dir)]
    else:
        logger = False
    
    args["mon"]["trainer"] |= {
        "callbacks"           : callbacks,
        "devices"             : devices,
        "default_root_dir"    : save_dir,
        "logger"              : logger,
        "max_epochs"          : epochs,
        "max_steps"           : steps,
        "num_sanity_val_steps": 0,
    }
    trainer               = mon.Trainer(**args["mon"]["trainer"])
    trainer.current_epoch = mon.get_epoch_from_checkpoint(ckpt=ckpt)
    trainer.global_step   = mon.get_global_step_from_checkpoint(ckpt=ckpt)
    if mon.is_rank_zero():
        console.log("[green]Done")
    
    # Training
    if mon.is_rank_zero():
        console.rule("[bold red] TRAINING")
    trainer.fit(
        model             = model,
        train_dataloaders = datamodule.train_dataloader,
        val_dataloaders   = datamodule.val_dataloader,
        ckpt_path         = ckpt,
    )
    if mon.is_rank_zero():
        console.log(f"Model: {fullname}")  # Log
        console.log("[green]Done")
    
    # Return
    return str(save_dir)
    
# endregion


# region Main

def main():
    args = mon.parse_train_args()
    train(args)


if __name__ == "__main__":
    main()

# endregion
