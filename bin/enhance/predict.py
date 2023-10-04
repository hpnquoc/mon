#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module implements prediction pipeline."""

from __future__ import annotations

import copy
import importlib
import socket
import time
from typing import Any

import click
import torch
import torchvision

import mon

console = mon.console


# region Host

hosts = {
	"lp-labdesktop-01": {
		"config"     : "",
        "root"       : mon.RUN_DIR / "predict",
        "project"    : None,
        "name"       : None,
        "variant"    : None,
        "weights"    : None,
        "batch_size" : 8,
        "image_size" : (512, 512),
        "accelerator": "auto",
		"devices"    : 0,
        "max_epochs" : None,
        "max_steps"  : None,
		"strategy"   : None,
	},
    "vsw-ws02": {
		"config"     : "",
        "root"       : mon.RUN_DIR / "predict",
        "project"    : None,
        "name"       : None,
        "variant"    : None,
        "weights"    : None,
        "batch_size" : 8,
        "image_size" : (512, 512),
        "accelerator": "auto",
		"devices"    : 0,
        "max_epochs" : None,
        "max_steps"  : None,
		"strategy"   : None,
	},
    "vsw-ws-03": {
		"config"     : "",
        "root"       : mon.RUN_DIR / "predict",
        "project"    : None,
        "name"       : None,
        "variant"    : None,
        "weights"    : None,
        "batch_size" : 8,
        "image_size" : (512, 512),
        "accelerator": "auto",
		"devices"    : 0,
        "max_epochs" : None,
        "max_steps"  : None,
		"strategy"   : None,
	},
}

# endregion


# region Function

def predict(args: dict):
    # Initialization
    model_name    = args["model"]["name"]
    variant       = args["model"]["variant"]
    model_variant = f"{model_name}-{variant}" if variant is not None else f"{model_name}"
    console.rule(f"[bold red] {model_variant}")
    
    weights          = args["model"]["weights"]
    model: mon.Model = mon.MODELS.build(config=args["model"])
    state_dict       = torch.load(weights)
    model.load_state_dict(state_dict=state_dict["state_dict"])
    model            = model.cuda()
    model.eval()
    
    output_dir = args["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Measure efficiency score
    flops, params, avg_time = mon.calculate_efficiency_score(
        model      = model,
        image_size = args["image_size"],
        channels   = 3,
        runs       = 100,
        use_cuda   = True,
        verbose    = False,
    )
    params = model.params
    console.log(f"FLOPs  = {flops:.4f}")
    console.log(f"Params = {params:.4f}")
    console.log(f"Time   = {avg_time:.4f}")
    
    data       = args["datamodule"]["root"]
    image_size = args["datamodule"]["image_size"]
    resize     = args["datamodule"]["resize"]
    console.log(f"{data}")
    
    #
    with torch.no_grad():
        image_paths = list(data.rglob("*"))
        image_paths = [path for path in image_paths if path.is_image_file()]
        image_paths.sort()
        h, w        = mon.get_hw(image_size)
        sum_time    = 0
        with mon.get_progress_bar() as pbar:
            for _, image_path in pbar.track(
                sequence    = enumerate(image_paths),
                total       = len(image_paths),
                description = f"[bright_yellow] Inferring"
            ):
                # console.log(image_path)
                image       = mon.read_image(path=image_path, to_rgb=True, to_tensor=True, normalize=True)
                if resize:
                    h0, w0  = mon.get_image_size(image)
                    image   = mon.resize(input=image, size=[h, w])
                input       = image.to(model.device)
                start_time  = time.time()
                output      = model(input=input, augment=False, profile=False, out_index=-1)
                run_time    = (time.time() - start_time)
                output      = output[-1] if isinstance(output, (list, tuple)) else output
                if resize:
                    output  = mon.resize(input=image, size=[h0, w0])
                result_path = output_dir / image_path.name
                torchvision.utils.save_image(output, str(result_path))
                sum_time   += run_time
        avg_time = float(sum_time / len(image_paths))
        console.log(f"Average time: {avg_time}")


@click.command(context_settings=dict(
    ignore_unknown_options = True,
    allow_extra_args       = True,
))
@click.option("--data",        default=mon.DATA_DIR,          type=click.Path(exists=True),  help="Source data directory.")
@click.option("--config",      default="",                    type=click.Path(exists=False), help="The training config to use.")
@click.option("--root",        default=mon.RUN_DIR/"predict", type=click.Path(exists=False), help="Save results to root/project/name.")
@click.option("--project",     default=None,                  type=click.Path(exists=False), help="Save results to root/project/name.")
@click.option("--name",        default=None,                  type=click.Path(exists=False), help="Save results to root/project/name.")
@click.option("--variant",     default=None,                  type=str,                      help="Model variant.")
@click.option("--weights",     default=None,                  type=click.Path(exists=False), help="Weights paths.")
@click.option("--batch-size",  default=1,                     type=int,                      help="Total Batch size for all GPUs.")
@click.option("--image-size",  default=512,                   type=int,                      help="Image sizes.")
@click.option("--resize",      is_flag=True)
@click.option("--output-dir",  default=mon.RUN_DIR/"predict", type=click.Path(exists=False), help="Save results to root/project/name.")
@click.option("--verbose",     is_flag=True)
@click.pass_context
def main(
    ctx,
    data       : mon.Path | str,
    config     : mon.Path | str,
    root       : mon.Path | str,
    project    : str,
    name       : str,
    variant    : int | str | None,
    weights    : Any,
    batch_size : int,
    image_size : int | list[int],
    resize     : bool,
    output_dir : mon.Path | str,
    verbose    : bool
):
    model_kwargs = {
        k.lstrip("--"): ctx.args[i + 1]
            if not (i + 1 >= len(ctx.args) or ctx.args[i + 1].startswith("--"))
            else True for i, k in enumerate(ctx.args) if k.startswith("--")
    }
    
    # Obtain arguments
    hostname  = socket.gethostname().lower()
    host_args = hosts[hostname]
    config    = config  or host_args.get("config",  None)
    project   = project or host_args.get("project", None)
    
    if project is not None and project != "":
        project_module = project.replace("/", ".")
        config_args    = importlib.import_module(f"config.{project_module}.{config}")
    else:
        config_args    = importlib.import_module(f"config.{config}")
    
    # Prioritize input args --> predefined args --> config file args
    data        = mon.Path(data)
    project     = project or config_args.model["project"]
    project     = str(project).replace(".", "/")
    root        = root        or host_args.get("root",        None)
    name        = name        or host_args.get("name",        None) or config_args.model["name"]
    variant     = variant     or host_args.get("variant",     None) or config_args.model["variant"]
    weights     = weights     or host_args.get("weights",     None) or config_args.model["weights"]
    batch_size  = batch_size  or host_args.get("batch_size",  None) or config_args.data["batch_size"]
    image_size  = image_size  or host_args.get("image_size",  None) or config_args.data["image_size"]
    
    # Update arguments
    args                 = mon.get_module_vars(config_args)
    args["hostname"]     = hostname
    args["root"]         = mon.Path(root)
    args["project"]      = project
    args["image_size"]   = image_size
    args["output_dir"]   = mon.Path(output_dir)
    args["config_file"]  = config_args.__file__,
    args["datamodule"]  |= {
        "root"      : data,
        "resize"    : resize,
        "image_size": image_size,
        "batch_size": batch_size,
    }
    args["model"] |= {
        "weights": weights,
        "name"   : name,
        "variant": variant,
        "root"   : root,
        "project": project,
        "verbose": verbose,
    }
    args["model"] |= model_kwargs
    
    predict(args=args)

# endregion


# region Main

if __name__ == "__main__":
    main()

# endregion
