#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements interactive CLI."""

__all__ = [
    "CLI_OPTIONS",
    "DEFAULT_ARGS",
    "parse_default_args",
    "parse_predict_args",
    "parse_train_args",
]

import argparse
import socket
from typing import Any

import box

from mon.core.device import list_devices, parse_device
from mon.core.dtypes import image as I
from mon.core.enum import Task, TRTPrecision
from mon.core.pathlib import Path
from mon.core.utils import merge_dicts
from .utils import (
    load_config,
    parse_config_file,
    parse_save_dir,
    parse_weights_file,
)


# ----- Utils -----
def _str_or_none(a_str: Any) -> str | None:
    """Converts a value to a ``str`` or ``None`` if value is ``"None"``.

    Args:
        a_str: Value to convert.

    Returns:
        A ``str`` or ``None``.
    """
    return None if a_str in [None, "None", ""] else str(a_str)


def _int_or_none(int_or_str: Any) -> int | None:
    """Converts a value to an ``int`` or ``None`` if value is ``"None"``.

    Args:
        int_or_str: Value to convert.

    Returns:
        An ``int`` or ``None``.
    """
    return None if int_or_str in [None, "None", ""] else int(int_or_str)


def _float_or_none(float_or_str: Any) -> float | None:
    """Converts a value to a float or ``None`` if value is ``"None"``.

    Args:
        float_or_str: Value to convert.

    Returns:
        A float or ``"None"``.
    """
    return None if float_or_str in [None, "None", ""] else float(float_or_str)


# ----- Default CLI Options -----
CLI_OPTIONS  = {
    # Basic
    "root"         : {
        "default"    : None,
        "type"       : _str_or_none,
        "help"       : "Project root.",
        "prompt_only": False,
        "prompt_text": "Project Root",
    },
    "task"         : {
        "default"    : None,
        "type"       : _str_or_none,
        "choices"    : Task.values(),
        "help"       : f"Task to run: {Task.values()}.",
        "prompt_only": False,
        "prompt_text": "Task",
    },
    "mode"         : {
        "default"    : None,
        "type"       : _str_or_none,
        "choices"    : ["train", "predict", "speed"],  # RunMode.values(),
        "help"       : f"Run mode: {['train', 'predict']}.",
        "prompt_only": False,
        "i_cli_type" : str,
        "prompt_text": "Run Mode",
    },
    "arch"         : {
        "default"    : None,
        "type"       : _str_or_none,
        "help"       : "Model architecture.",
        "prompt_only": False,
        "prompt_text": "Architecture",
    },
    "model"        : {
        "default"    : None,
        "type"       : _str_or_none,
        "help"       : "Model name.",
        "prompt_only": False,
        "prompt_text": "Model",
    },
    "config"       : {
        "default"    : None,
        "type"       : _str_or_none,
        "help"       : "Config file.",
        "prompt_only": False,
        "prompt_text": "Config",
    },
    "data"         : {
        "default"    : None,
        "type"       : _str_or_none,
        "help"       : "Dataset name or directory.",
        "prompt_only": False,
        "prompt_text": "Predict(s)",
    },
    "fullname"     : {
        "default"    : None,
        "type"       : _str_or_none,
        "help"       : "Full name of the current run.",
        "prompt_only": False,  
        "prompt_text": "Fullname",
    },
    "save_dir"     : {
        "type"       : _str_or_none,
        "default"    : None,
        "help"       : "Directory to save the outputs.",
        "prompt_only": False,
        "prompt_text": "Save Directory",
    },
    "weights"      : {
        "action"     : "append",
        "default"    : None,
        "type"       : _str_or_none,
        "help"       : "Path(s) to the pretrained weights.",
        "prompt_only": False,
        "prompt_text": "Weights",
    },
    "device"       : {
        "default"    : None,
        "type"       : _str_or_none,
        "choices"    : list_devices(),
        "help"       : f"Running device: {list_devices()}.",
        "prompt_only": False,
        "prompt_text": "Device",
    },
    "seed"         : {
        "default"    : None,
        "type"       : _int_or_none,
        "help"       : "Seed.",
        "prompt_only": False,
        "prompt_text": "Seed         ",
    },
    "imgsz"        : {
        "action"     : "append",
        "default"    : None,
        "type"       : _int_or_none,
        "help"       : "Image size.",
        "prompt_only": False,
        "prompt_text": "Image Size   ",
    },
    # Train
    "epochs"       : {
        "default"    : None,
        "type"       : _int_or_none,
        "help"       : "Training epochs.",
        "prompt_only": False,
        "prompt_text": "Epochs       ",
    },
    "batch_size"   : {
        "default"    : None,
        "type"       : _int_or_none,
        "help"       : "Batch size.",
        "prompt_only": False,
        "prompt_text": "Batch Size   ",
    },
    "torchrun"     : {
        "default"    : False,
        "action"     : "store_true",
        "help"       : "Using torch distributed training.",
        "prompt_only": False,
        "prompt_text": "Use torchrun?",
    },
    "master_port"  : {
        "default"    : 7777,
        "type"       : _int_or_none,
        "help"       : "Port for distributed communication.",
        "prompt_only": False,
        "prompt_text": "Master Port",
    },
    "master_addr"  : {
        "default"    : "localhost",
        "type"       : _str_or_none,
        "help"       : "Master node address.",
        "prompt_only": False,
        "prompt_text": "Master Address",
    },
    "local_rank"   : {
        "type"       : _int_or_none,
        "help"       : "Local rank for distributed training.",
        "prompt_only": False,
        "prompt_text": "Local Rank   ",
    },
    # Predict
    "resize"       : {
        "action"     : "store_true",
        "help"       : "Resize the input image.",
        "prompt_only": False,
        "prompt_text": "Resize?      ",
    },
    "benchmark"    : {
        "action"     : "store_true",
        "help"       : "Enable benchmark mode.",
        "prompt_only": False,
        "prompt_text": "Benchmark?   ",
    },
    # Save & Visualize
    "save_result"  : {
        "action"     : "store_true",
        "help"       : "Save results.",
        "prompt_only": False,
        "prompt_text": "Save Result? ",
    },
    "save_image"   : {
        "action"     : "store_true",
        "help"       : "Save output images.",
        "prompt_only": False,
        "prompt_text": "Save Image?  ",
    },
    "save_debug"   : {
        "action"     : "store_true",
        "help"       : "Save debug information.",
        "prompt_only": False,
        "prompt_text": "Save Debug?  ",
    },
    "use_fullname" : {
        "action"     : "store_true",
        "help"       : "Use the ``fullname`` for the ``save_dir``.",
        "prompt_only": False,
        "prompt_text": "Use Fullname?",
    },
    "keep_subdirs" : {
        "action"     : "store_true",
        "help"       : "Keep subdirectories in the ``save_dir``.",
        "prompt_only": False,
        "prompt_text": "Keep Subdirs?",
    },
    "save_nearby"  : {
        "action"     : "store_true",
        "help"       : "Save outputs nearby the source.",
        "prompt_only": False,
        "prompt_text": "Save Nearby? ",
    },
    "exist_ok"     : {
        "action"     : "store_true",
        "help"       : "Keep existing directories.",
        "prompt_only": False,
        "prompt_text": "Exist OK?    ",
    },
    "verbose"      : {
        "action"     : "store_true",
        "help"       : "Verbose mode.",
        "prompt_only": False,
        "prompt_text": "Verbosity?   ",
    },
    # Export
    "trt_precision": {
        "default"    : "fp32",
        "type"       : _str_or_none,
        "choices"    : TRTPrecision.values(),
        "help"       : f"TRT precision: {TRTPrecision.values()}.",
        "prompt_only": False,
        "prompt_text": "TRT Precision",
    },
}
CLI_OPTIONS  = box.Box(CLI_OPTIONS)

DEFAULT_ARGS = {
    k: False if v.get("action") in ["store_true"] else v.get("default", None)
    for k, v in CLI_OPTIONS.items()
}
DEFAULT_ARGS = box.Box(DEFAULT_ARGS)


# ----- Parser -----
def parse_default_args(name: str = "main") -> dict | box.Box:
    """Parse default arguments."""
    parser = argparse.ArgumentParser(description=name)
    
    for opt_name, opt_params in CLI_OPTIONS.items():
        action      = opt_params.get("action",      "store")
        default     = opt_params.get("default",     None)
        opt_type    = opt_params.get("type",        None)
        choices     = opt_params.get("choices",     None)
        required    = opt_params.get("required",    False)
        help_text   = opt_params.get("help",        "")
        prompt_only = opt_params.get("prompt_only", False)  # Use in interactive CLI only, not parse_args
        
        if prompt_only:
            continue
        '''
        if opt_type == bool and default is None:
            default = False
        if action == "store_true" and default is None:
            default = False
        if action == "store_false" and default is None:
            default = True
        '''
        
        kwargs = {
            "action"  : action,
            "default" : default,
            "required": required,
            "help"    : help_text,
        }
        if action in ["store_true", "store_false"]:
            kwargs.pop("default")
            # kwargs["default"] = False if action == "store_true" else True
        if opt_type:
            kwargs["type"] = opt_type
        if choices:
            kwargs["choices"] = choices
        flag = f"--{opt_name.replace('_', '-')}"
        parser.add_argument(flag, **kwargs)

    parser.add_argument("extra_args", nargs=argparse.REMAINDER, help="Additional arguments")
    return box.Box(vars(parser.parse_args()))


def parse_train_args(model_root: Path = None, verbose: bool = False) -> dict | box.Box:
    """Parse arguments for training."""
    # Get args
    cli        = parse_default_args()
    cli.root   = Path(cli.root) if cli.root else None
    cli.config = parse_config_file(cli.config, cli.root, model_root=model_root)
    args       = load_config(cli.config, verbose=verbose)
    args       = merge_dicts(args, cli)  # Prioritize cli -> args
    
    # Parse arguments
    if args.save_dir in [None, ""]:
        if args.use_fullname:
            args.save_dir = parse_save_dir(args.root/"run"/"train", args.arch, args.model, args.fullname)
            # args.save_dir = _utils.parse_save_dir(root/"run"/"train", arch, fullname, None)
        else:
            args.save_dir = parse_save_dir(args.root/"run"/"train", args.arch, args.model, args.data)
    else:
        args.save_dir = Path(args.save_dir)
        # if str("run/train") not in str(args.save_dir):
        #     args.save_dir = pathlib.Path(f"run/train/{args.save_dir}")
        # if str(args.root) not in str(args.save_dir):
        #     args.save_dir = args.root / args.save_dir

    args.hostname = socket.gethostname().lower()
    args.weights  = parse_weights_file(args.root, args.weights)
    args.resume   = parse_weights_file(args.root, args.resume)
    args.tuning   = parse_weights_file(args.root, args.tuning)
    args.device   = parse_device(args.device)

    # Save config file
    if not args.exist_ok:
        args.save_dir.rmdir()
    args.save_dir.mkdir(parents=True, exist_ok=True)
    if args.config and args.config.is_config_file():
        args.config.copy_to(dst=args.save_dir / f"{args.config.name}")

    return args


def parse_predict_args(model_root: Path = None, verbose: bool = False) -> dict | box.Box:
    """Parse arguments for predicting."""
    # Get args
    cli        = parse_default_args()
    cli.root   = Path(cli.root) if cli.root else None
    cli.config = parse_config_file(cli.config, cli.root, model_root=model_root)
    args       = load_config(cli.config, verbose=verbose)
    args       = merge_dicts(args, cli)  # Prioritize cli -> args
    
    # Parse arguments
    if args.save_dir in [None, ""]:
        if args.use_fullname or args.save_nearby:
            args.save_dir = parse_save_dir(args.root/"run"/"predict", args.arch, args.fullname, None)
        else:
            args.save_dir = parse_save_dir(args.root/"run"/"predict", args.arch, args.model, args.data)
    else:
        args.save_dir = Path(args.save_dir)
        # args.save_dir = args.save_dir.replace("run/train/", "")
        # if str("run/predict") not in str(args.save_dir):
        #     args.save_dir = pathlib.Path(f"run/predict/{args.save_dir}")
        # if str(args.root) not in str(args.save_dir):
        #     args.save_dir = args.root / args.save_dir
    
    args.hostname = socket.gethostname().lower()
    args.weights  = parse_weights_file(args.root, args.weights)
    args.resume   = parse_weights_file(args.root, args.resume)
    args.tuning   = parse_weights_file(args.root, args.tuning)
    args.device   = parse_device(args.device)
    args.imgsz    = I.imgsz(args.imgsz)
    
    # Save config file
    if not args.exist_ok:
        args.save_dir.rmdir()
    if not args.save_nearby and (args.save_result or args.save_image or args.save_debug):
        args.save_dir.mkdir(parents=True, exist_ok=True)
        if args.config and args.config.is_config_file():
            args.config.copy_to(dst=args.save_dir / f"{args.config.name}")

    return args
