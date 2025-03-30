#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Parses user-input and config arguments."""

from __future__ import annotations

__all__ = [
    "parse_cli_args",
    "parse_predict_args",
    "parse_train_args",
]

import argparse
import socket
from typing import Any

from mon.core import pathlib, utils


# region Utils

def _str_or_none(value: Any) -> str | None:
    """Converts a value to a string or ``None`` if value is ``"None"``.

    Args:
        value: Value to convert.

    Returns:
        String of ``value`` or ``None`` if ``value`` is ``"None"``.
    """
    return None if value == "None" else str(value)


def _int_or_none(value: Any) -> int | None:
    """Converts a value to an integer or ``None`` if value is ``"None"``.

    Args:
        value: Value to convert.

    Returns:
        Integer of ``value`` or ``None`` if ``value`` is ``"None"``.
    """
    return None if value == "None" else int(value)


def _float_or_none(value: Any) -> float | None:
    """Converts a value to a float or ``None`` if value is ``"None"``.

    Args:
        value: Value to convert.

    Returns:
        Float of ``value`` or ``None`` if ``value`` is ``"None"``.
    """
    return None if value == "None" else float(value)


def get_image_size(input: Any) -> tuple[int, int]:
    """Retrieves the size of an image as a width-height tuple.

    Args:
        input: Image input to measure.

    Returns:
        Tuple of ``(width, height)`` in integers.
    """
    from mon.vision import get_image_size
    return get_image_size(input)

# endregion


def parse_cli_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="predict")
    # Basic
    parser.add_argument("--config",       type=_str_or_none, default=None, help="Model config.")
    parser.add_argument("--root",         type=_str_or_none, default=None, help="Root directory of the current run.")
    parser.add_argument("--arch",         type=_str_or_none, default=None, help="Model architecture or family.")
    parser.add_argument("--model",        type=_str_or_none, default=None, help="Model name.")
    parser.add_argument("--data",         type=_str_or_none, default=None, help="Dataset name or directory.")
    parser.add_argument("--fullname",     type=_str_or_none, default=None, help="Full name of the current run.")
    parser.add_argument("--save-dir",     type=_str_or_none, default=None, help="Saving directory. If not set, it will be `root/arch/model/data`.")
    parser.add_argument("--weights",      type=_str_or_none, default=None, help="Weights paths.")
    parser.add_argument("--device",       type=_str_or_none, default=None, help="Running devices.")
    parser.add_argument("--imgsz",        type=_int_or_none, default=None, help="Image sizes.")
    parser.add_argument("--resize",       action="store_true",             help="Resize the input image to `imgsz`.")
    parser.add_argument("--epochs",       type=_int_or_none, default=None, help="Training epochs.")
    parser.add_argument("--steps",        type=_int_or_none, default=None, help="Training steps.")
    parser.add_argument("--benchmark",    action="store_true",             help="Benchmark the model.")
    parser.add_argument("--save-image",   action="store_true",             help="Save the output image.")
    parser.add_argument("--save-debug",   action="store_true",             help="Save the debug information.")
    parser.add_argument("--use-fullname", action="store_true",             help="Use the full name for the save_dir.")
    parser.add_argument("--use-fullpath", action="store_true",             help="Use the full path for the input image.")
    parser.add_argument("--exist-ok",     action="store_true",             help="If ``False``, it will delete the save directory if it already exists.")
    parser.add_argument("--verbose",      action="store_true")
    parser.add_argument("extra_args",     nargs=argparse.REMAINDER,        help="Additional arguments")
    args = parser.parse_args()
    return args


def parse_train_args(model_root: str | pathlib.Path = None) -> dict | argparse.Namespace:
    """Parse arguments for training."""
    hostname = socket.gethostname().lower()
    
    # Get input args
    cli_args = vars(parse_cli_args())
    config   = cli_args.get("config")
    root     = cli_args.get("root")
    root     = pathlib.Path(root) if root else None
    weights  = cli_args.get("weights")
    
    # Get config args
    config = utils.parse_config_file(
        project_root = root,
        model_root   = model_root,
        weights_path = weights,
        config       = config,
    )
    args   = utils.load_config(config)
    
    # Prioritize cli_args -> args
    root         = root                         or args.get("root")
    arch         = cli_args.get("arch")         or args.get("arch")
    model        = cli_args.get("model")        or args.get("model")
    data         = cli_args.get("data")         or args.get("data")
    fullname     = cli_args.get("fullname")     or args.get("fullname")
    save_dir     = cli_args.get("save_dir")     or args.get("save_dir")
    weights      = cli_args.get("weights")      or args.get("weights")
    device       = cli_args.get("device")       or args.get("device")
    imgsz        = cli_args.get("imgsz")        or args.get("imgsz")
    resize       = cli_args.get("resize")       or args.get("resize")
    epochs       = cli_args.get("epochs")       or args.get("epochs")
    steps        = cli_args.get("steps")        or args.get("steps")
    benchmark    = cli_args.get("benchmark")    or args.get("benchmark")
    save_image   = cli_args.get("save_image")   or args.get("save_image")
    save_debug   = cli_args.get("save_debug")   or args.get("save_debug")
    use_fullname =                                 args.get("use_fullname", False)
    use_fullpath = cli_args.get("use_fullpath") or args.get("use_fullpath")
    exist_ok     = cli_args.get("exist_ok")     or args.get("exist_ok")
    verbose      = cli_args.get("verbose")      or args.get("verbose")
    extra_args   = cli_args.get("extra_args")
    
    # Parse arguments
    if save_dir in [None, ""]:
        save_dir = utils.parse_save_dir(root/"run"/"train", arch, model, data)
    else:
        save_dir = pathlib.Path(save_dir)
        if str("run/train") not in str(save_dir):
            save_dir = pathlib.Path(f"run/train/{save_dir}")
        if str(root) not in str(save_dir):
            save_dir = root / save_dir
            
    weights = utils.parse_weights_file(root/"run"/"train", weights)
    device  = utils.parse_device(device)
    
    # Update arguments
    args["hostname"]     = hostname
    args["root"]         = root
    args["arch"]         = arch
    args["model"]        = model
    args["data"]         = data
    args["fullname"]     = fullname
    args["save_dir"]     = save_dir
    args["weights"]      = weights
    args["device"]       = device
    args["imgsz"]        = imgsz
    args["resize"]       = resize
    args["epochs"]       = epochs
    args["steps"]        = steps
    args["benchmark"]    = benchmark
    args["save_image"]   = save_image
    args["save_debug"]   = save_debug
    args["use_fullname"] = use_fullname
    args["use_fullpath"] = use_fullpath
    args["exist_ok"]     = exist_ok
    args["verbose"]      = verbose
    args |= extra_args
    
    # Save config file
    if not exist_ok:
        pathlib.delete_dir(paths=save_dir)
        
    save_dir.mkdir(parents=True, exist_ok=True)
    if config and config.is_config_file():
        # pathlib.copy_file(src=config, dst=save_dir / f"config{config.suffix}")
        pathlib.copy_file(src=config, dst=save_dir / f"{config.name}")
    
    # Return
    # args = argparse.Namespace(**args)
    return args


def parse_predict_args(model_root: str | pathlib.Path = None) -> dict | argparse.Namespace:
    """Parse arguments for predicting."""
    hostname = socket.gethostname().lower()
    
    # Get input args
    cli_args = vars(parse_cli_args())
    config   = cli_args.get("config")
    root     = cli_args.get("root")
    root     = pathlib.Path(root) if root else None
    weights  = cli_args.get("weights")
    
    # Get config args
    config = utils.parse_config_file(
        project_root = root,
        model_root   = model_root,
        weights_path = weights,
        config       = config,
    )
    args   = utils.load_config(config)
    
    # Prioritize cli_args -> args
    root         = root                         or args.get("root")
    arch         = cli_args.get("arch")         or args.get("arch")
    model        = cli_args.get("model")        or args.get("model")
    data         = cli_args.get("data")         or args.get("data")
    fullname     = cli_args.get("fullname")     or args.get("fullname")
    save_dir     = cli_args.get("save_dir")     or args.get("save_dir")
    weights      = cli_args.get("weights")      or args.get("weights")
    device       = cli_args.get("device")       or args.get("device")
    imgsz        = cli_args.get("imgsz")        or args.get("imgsz")
    resize       = cli_args.get("resize")       or args.get("resize")
    epochs       = cli_args.get("epochs")       or args.get("epochs")
    steps        = cli_args.get("steps")        or args.get("steps")
    benchmark    = cli_args.get("benchmark")    or args.get("benchmark")
    save_image   = cli_args.get("save_image")   or args.get("save_image")
    save_debug   = cli_args.get("save_debug")   or args.get("save_debug")
    use_fullname =                                 args.get("use_fullname", False)
    use_fullpath = cli_args.get("use_fullpath") or args.get("use_fullpath")
    exist_ok     = cli_args.get("exist_ok")     or args.get("exist_ok")
    verbose      = cli_args.get("verbose")      or args.get("verbose")
    extra_args   = cli_args.get("extra_args")
    
    # Parse arguments
    if save_dir in [None, ""]:
        if use_fullname:
            save_dir = utils.parse_save_dir(root/"run"/"predict", arch, fullname, None)
        else:
            save_dir = utils.parse_save_dir(root/"run"/"predict", arch, model,    None)
    else:
        save_dir = pathlib.Path(save_dir)
        save_dir = save_dir.replace("run/train/", "")
        if str("run/predict") not in str(save_dir):
            save_dir = pathlib.Path(f"run/predict/{save_dir}")
        if str(root) not in str(save_dir):
            save_dir = root / save_dir
        
    weights = utils.parse_weights_file(root, weights)
    device  = utils.parse_device(device)
    imgsz   = get_image_size(imgsz)
    
    # Update arguments
    args["hostname"]     = hostname
    args["root"]         = root
    args["arch"]         = arch
    args["model"]        = model
    args["data"]         = data
    args["fullname"]     = fullname
    args["save_dir"]     = save_dir
    args["weights"]      = weights
    args["device"]       = device
    args["imgsz"]        = imgsz
    args["resize"]       = resize
    args["epochs"]       = epochs
    args["steps"]        = steps
    args["benchmark"]    = benchmark
    args["save_image"]   = save_image
    args["save_debug"]   = save_debug
    args["use_fullname"] = use_fullname
    args["use_fullpath"] = use_fullpath
    args["exist_ok"]     = exist_ok
    args["verbose"]      = verbose
    args |= extra_args
    
    # Save config file
    if not exist_ok:
        pathlib.delete_dir(paths=save_dir)
    
    save_dir.mkdir(parents=True, exist_ok=True)
    if config and config.is_config_file():
        # pathlib.copy_file(src=config, dst=save_dir / f"config{config.suffix}")
        pathlib.copy_file(src=config, dst=save_dir / f"{config.name}")
    
    # Return
    # args = argparse.Namespace(**args)
    return args
