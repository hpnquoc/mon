#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Trains a model on a given dataset."""

import box

import mon

current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# ----- Train -----
def train(args: dict | box.Box) -> str:
    # Start
    if mon.is_rank_zero():
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
    args["modelmodule"] |= {
        "fullname": args.fullname,
        "root"    : args.save_dir,
        "weights" : pretrained,
        "debug"   : args.save_debug,
        "verbose" : args.verbose,
    }
    model: mon.LightningModule = mon.MODELS.build(config=args.modelmodule)
    if mon.is_rank_zero():
        mon.print_dict(args, title=args.fullname)

    # Trainer
    callbacks = args.trainer.callbacks
    for i, callback in enumerate(callbacks):
        if callback.name == "model_checkpoint":
            callbacks[i] |= {"filename": args.fullname}
    callbacks = mon.CALLBACKS.build_instances(configs=args.trainer.callbacks)
    ckpt      = mon.get_latest_checkpoint(dirpath=model.ckpt_dir)
    devices   = mon.to_int_list(device) if "auto" not in device else "auto"
    if args.trainer.logger:
        logger = [mon.TensorBoardLogger(save_dir=args.save_dir)]
    else:
        logger = False
    
    args["trainer"] |= {
        "callbacks"           : callbacks,
        "devices"             : devices,
        "default_root_dir"    : args.save_dir,
        "logger"              : logger,
        "max_epochs"          : args.epochs,
        "num_sanity_val_steps": 0,
    }
    trainer               = mon.Trainer(**args.trainer)
    trainer.current_epoch = mon.get_epoch_from_checkpoint(ckpt=ckpt)
    trainer.global_step   = mon.get_global_step_from_checkpoint(ckpt=ckpt)

    # Train
    if mon.is_rank_zero():
        mon.console.rule("[bold red] TRAINING")
    trainer.fit(
        model             = model,
        train_dataloaders = datamodule.train_dataloader,
        val_dataloaders   = datamodule.val_dataloader,
        ckpt_path         = ckpt,
    )

    # Finish
    return str(args.save_dir)
    

# ----- Main -----
def main():
    args = mon.parse_train_args()
    train(args)


if __name__ == "__main__":
    main()
