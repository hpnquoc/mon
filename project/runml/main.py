#!/usr/bin/env python
# -*- coding: utf-8 -*-

""""""

from __future__ import annotations

import subprocess

import click

import mon

current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]
_modes 	      = ["train", "predict", "online", "instance", "metric"]


# region Install

def run_install(args: dict):
    # Get user input
    root  = mon.Path(args["root"])
    model = args["model"]
    
    assert root.exists()
    
    # Parse arguments
    use_extra_model = mon.is_extra_model(model)
    model           = mon.parse_model_name(model)
    
    # Parse script file
    if use_extra_model:
        requirement_file = mon.EXTRA_MODELS[model]["model_dir"] / "requirements.txt"
        if requirement_file.is_txt_file():
            result = subprocess.run(["pip", "install", "-r", f"{str(requirement_file)}"], cwd=current_dir)
            print(result)
            
# endregion


# region Train

def run_train(args: dict):
    # Get user input
    task     = args["task"]
    mode     = args["mode"]
    config   = args["config"]
    arch     = args["arch"]
    model    = args["model"]
    root     = mon.Path(args["root"])
    project  = args["project"]
    variant  = args["variant"]
    fullname = args["fullname"]
    save_dir = args["save_dir"]
    weights  = args["weights"]
    device   = args["device"]
    epochs   = args["epochs"]
    steps    = args["steps"]
    exist_ok = args["exist_ok"]
    verbose  = args["verbose"]
    
    assert root.exists()
    
    # Parse arguments
    use_extra_model = mon.is_extra_model(model)
    model    = mon.parse_model_name(model)
    fullname = fullname if fullname not in [None, "None", ""] else config.stem
    config   = mon.parse_config_file(
        project_root = root,
        model_root   = mon.EXTRA_MODELS[arch][model]["model_dir"] if use_extra_model else None,
        weights_path = weights,
        config       = config,
    )
    assert config not in [None, "None", ""]
    # save_dir = save_dir or mon.parse_save_dir(root/"run"/"train", arch, model, data, project, variant)
    weights  = mon.to_str(weights, ",")
    
    kwargs   = {
        "--config"  : config,
        "--arch"    : arch,
        "--model"   : model,
        "--root"    : str(root),
        "--project" : project,
        "--variant" : variant,
        "--fullname": fullname,
        "--save-dir": str(save_dir),
        "--weights" : weights,
        "--device"  : device,
        "--epochs"  : epochs,
        "--steps"   : steps,
    }
    flags    = ["--exist-ok"] if exist_ok else []
    flags   += ["--verbose"]  if verbose  else []
    
    # Parse script file
    if use_extra_model:
        torch_distributed_launch = mon.EXTRA_MODELS[arch][model]["torch_distributed_launch"]
        script_file = mon.EXTRA_MODELS[arch][model]["model_dir"] / "my_train.py"
        device      = mon.parse_device(device)
        if isinstance(device, list) and torch_distributed_launch:
            python_call = [
                f"python",
                f"-m",
                f"torch.distributed.launch",
                f"--nproc_per_node={str(len(device))}",
                f"--master_port=9527"
            ]
        else:
            python_call = ["python"]
    else:
        script_file = current_dir / "train.py"
        python_call = ["python"]
    
    # Parse arguments
    args_call: list[str] = []
    for k, v in kwargs.items():
        if v is None:
            continue
        elif isinstance(v, list | tuple):
            args_call_ = [f"{k}={v_}" for v_ in v]
        else:
            args_call_ = [f"{k}={v}"]
        args_call += args_call_
    
    # Run training
    if script_file.is_py_file():
        print("\n")
        command = (
            python_call +
            [script_file] +
            args_call +
            flags
        )
        result = subprocess.run(command, cwd=current_dir)
        print(result)
    else:
        raise ValueError(f"Cannot find Python training script file at: {script_file}.")
    
# endregion


# region Predict

def run_predict(args: dict):
    # Get user input
    task       = args["task"]
    mode       = args["mode"]
    config     = args["config"]
    arch       = args["arch"]
    model      = args["model"]
    data       = args["data"]
    root       = mon.Path(args["root"])
    project    = args["project"]
    variant    = args["variant"]
    fullname   = args["fullname"]
    save_dir   = args["save_dir"]
    weights    = args["weights"]
    device     = args["device"]
    imgsz      = args["imgsz"]
    resize     = args["resize"]
    benchmark  = args["benchmark"]
    save_image = args["save_image"]
    save_debug = args["save_debug"]
    verbose    = args["verbose"]
    
    assert root.exists()
    
    # Parse arguments
    use_extra_model = mon.is_extra_model(model)
    model    = mon.parse_model_name(model)
    fullname = fullname if fullname not in [None, "None", ""] else model
    config   = mon.parse_config_file(
        project_root = root,
        model_root   = mon.EXTRA_MODELS[arch][model]["model_dir"] if use_extra_model else None,
        weights_path = weights,
        config       = config,
    )
    config   = config or "default"
    # if use_data_dir:
    #     save_dir = save_dir or mon.parse_save_dir(mon.DATA_DIR/task.value/"#predict", arch, model, None, project, variant)
    # else:
    #     save_dir = save_dir or mon.parse_save_dir(root/"run"/"predict", arch, model, None, project, variant)
    weights  = mon.to_str(weights, ",")
    
    for d in data:
        kwargs  = {
            "--config"  : config,
            "--arch"    : arch,
            "--model"   : model,
            "--data"    : d,
            "--root"    : str(root),
            "--project" : project,
            "--variant" : variant,
            "--fullname": fullname,
            "--save-dir": str(save_dir),
            "--weights" : weights,
            "--device"  : device,
            "--imgsz"   : imgsz,
        }
        flags   = ["--resize"]     if resize     else []
        flags  += ["--benchmark"]  if benchmark  else []
        flags  += ["--save-image"] if save_image else []
        flags  += ["--save-debug"] if save_debug else []
        flags  += ["--verbose"]    if verbose    else []
        
        # Parse script file
        if use_extra_model:
            torch_distributed_launch = mon.EXTRA_MODELS[arch][model]["torch_distributed_launch"]
            script_file = mon.EXTRA_MODELS[arch][model]["model_dir"] / "my_predict.py"
            python_call = ["python"]
        else:
            script_file = current_dir / "predict.py"
            python_call = ["python"]
        
        # Parse arguments
        args_call: list[str] = []
        for k, v in kwargs.items():
            if v is None:
                continue
            elif isinstance(v, list | tuple):
                args_call_ = [f"{k}={v_}" for v_ in v]
            else:
                args_call_ = [f"{k}={v}"]
            args_call += args_call_
        
        # Run prediction
        if script_file.is_py_file():
            print("\n")
            command = (
                python_call +
                [script_file] +
                args_call +
                flags
            )
            result = subprocess.run(command, cwd=current_dir)
            print(result)
        else:
            raise ValueError(f"Cannot find Python predicting script file at: {script_file}.")
        
# endregion


# region Online

def run_online(args: dict):
    # Get user input
    task         = args["task"]
    mode         = args["mode"]
    config       = args["config"]
    arch         = args["arch"]
    model        = args["model"]
    data         = args["data"]
    root         = mon.Path(args["root"])
    project      = args["project"]
    variant      = args["variant"]
    fullname     = args["fullname"]
    save_dir     = args["save_dir"]
    weights	     = args["weights"]
    device       = args["device"]
    epochs       = args["epochs"]
    steps        = args["steps"]
    imgsz        = args["imgsz"]
    resize       = args["resize"]
    benchmark    = args["benchmark"]
    save_image   = args["save_image"]
    save_debug   = args["save_debug"]
    use_data_dir = args["use_data_dir"]
    verbose      = args["verbose"]
    
    assert root.exists()
    
    # Parse arguments
    use_extra_model = mon.is_extra_model(model)
    model    = mon.parse_model_name(model)
    fullname = fullname if fullname not in [None, "None", ""] else config.stem
    config   = mon.parse_config_file(
        project_root = root,
        model_root   = mon.EXTRA_MODELS[arch][model]["model_dir"] if use_extra_model else None,
        weights_path = weights,
        config       = config,
    )
    assert config not in [None, "None", ""]
    weights  = mon.to_str(weights, ",")
    
    for d in data:
        if use_data_dir:
            save_dir = save_dir or mon.DATA_DIR / task.value / "#predict" / model
        else:
            save_dir = save_dir or root / "run" / "predict" / model
        kwargs  = {
            "--config"  : config,
            "--arch"    : arch,
            "--model"   : model,
            "--data"    : d,
            "--root"    : str(root),
            "--project" : project,
            "--variant" : variant,
            "--fullname": fullname,
            "--save-dir": str(save_dir),
            "--weights" : weights,
            "--device"  : device,
            "--imgsz"   : imgsz,
        }
        flags   = ["--resize"]     if resize     else []
        flags  += ["--benchmark"]  if benchmark  else []
        flags  += ["--save-image"] if save_image else []
        flags  += ["--save-debug"] if save_debug else []
        flags  += ["--verbose"]    if verbose    else []
        
        # Parse script file
        if use_extra_model:
            torch_distributed_launch = mon.EXTRA_MODELS[arch][model]["torch_distributed_launch"]
            script_file = mon.EXTRA_MODELS[arch][model]["model_dir"] / "my_online.py"
            python_call = ["python"]
        else:
            script_file = current_dir / "online.py"
            python_call = ["python"]
        
        # Parse arguments
        args_call: list[str] = []
        for k, v in kwargs.items():
            if v is None:
                continue
            elif isinstance(v, list | tuple):
                args_call_ = [f"{k}={v_}" for v_ in v]
            else:
                args_call_ = [f"{k}={v}"]
            args_call += args_call_
        
        # Run prediction
        if script_file.is_py_file():
            print("\n")
            command = (
                python_call +
                [script_file] +
                args_call +
                flags
            )
            result = subprocess.run(command, cwd=current_dir)
            print(result)
        else:
            raise ValueError(f"Cannot find Python predicting script file at: {script_file}.")

# endregion


# region Main

@click.command(name="main", context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.option("--root",     type=click.Path(exists=True), help="Project root.")
@click.option("--task",     type=str, default=None,       help="Running task.")
@click.option("--mode",     type=str, default="predict",  help="Running mode.")
@click.option("--arch",     type=str, default=None,       help="Running architecture.")
@click.option("--model",    type=str, default=None,       help="Running model.")
@click.option("--config",   type=str, default=None,   	  help="Running config.")
@click.option("--data",     type=str, default=None,       help="Predict dataset.")
@click.option("--project",  type=str, default=None,       help="Project name.")
@click.option("--variant",  type=str, default=None,       help="Variant name.")
@click.option("--save-dir", type=str, default=None,       help="Optional saving directory.")
@click.option("--weights",  type=str, default=None,       help="Weights paths.")
@click.option("--device",   type=str, default=None,       help="Running devices.")
@click.option("--epochs",   type=int, default=-1,   	  help="Training epochs.")
@click.option("--steps",    type=int, default=-1,   	  help="Training steps.")
@click.option("--imgsz",    type=int, default=-1,         help="Image size.")
@click.option("--exist-ok", is_flag=True,                 help="Exist OK.")
@click.option("--verbose",  is_flag=True,                 help="Verbosity.")
def main(
    root    : str,
    task    : str,
    mode    : str,
    arch    : str,
    model   : str,
    config  : str,
    data    : str,
    project : str,
    variant : str,
    save_dir: str,
    weights : str,
    device  : int | list[int] | str,
    epochs  : int,
    steps   : int,
    imgsz   : int,
    exist_ok: bool,
    verbose : bool,
):
    click.echo(click.style(f"\nInput Prompt:", fg="white", bg="red", bold=True))
    
    # Task
    tasks_     = mon.list_tasks(project_root=root)
    tasks_str_ = mon.parse_menu_string(tasks_)
    task       = click.prompt(click.style(f"Task {tasks_str_}", fg="bright_green", bold=True), default=task)
    task       = tasks_[int(task)] if mon.is_int(task) else task
    # Mode
    mode       = click.prompt(click.style(f"Mode {mon.parse_menu_string(_modes)}", fg="bright_green", bold=True), default=mode)
    mode       = _modes[int(mode)] if mon.is_int(mode) else mode
    # Architecture
    archs_       = mon.list_archs(project_root=root, task=task, mode=mode)
    archs_str_   = mon.parse_menu_string(archs_)
    arch	     = click.prompt(click.style(f"Architecture {archs_str_}", fg="bright_green", bold=True), type=str, default=arch)
    arch 	     = archs_[int(arch)] if mon.is_int(arch) else arch
    # Model
    models_      = mon.list_models(project_root=root, task=task, mode=mode, arch=arch)
    models_str_  = mon.parse_menu_string(models_)
    model	     = click.prompt(click.style(f"Model {models_str_}", fg="bright_green", bold=True), type=str, default=model)
    model 	     = models_[int(model)] if mon.is_int(model) else model
    model_name   = mon.parse_model_name(model)
    # Config
    model_dir    = mon.EXTRA_MODELS[arch][model_name]["model_dir"] if mon.is_extra_model(model) else None
    configs_     = mon.list_configs(project_root=root, model_root=model_dir, model=model)
    configs_str_ = mon.parse_menu_string(configs_)
    config	     = click.prompt(click.style(f"Config {configs_str_}", fg="bright_green", bold=True), type=str, default="")
    config       = configs_[int(config)] if mon.is_int(config) else config
    # Project
    project      = project if project not in [None, "None", ""] else ""
    project      = click.prompt(click.style(f"Project: {project}", fg="bright_green", bold=True), type=str, default=project)
    project      = None if project in [None, "None", ""] else project
    # Variant
    variant      = variant if variant not in [None, "None", ""] else ""
    variant      = click.prompt(click.style(f"Variant: {variant}", fg="bright_green", bold=True), type=str, default=variant)
    variant      = None if variant in [None, "None", ""] else variant
    # Weights
    weights_     = mon.list_weights_files(project_root=root, model=model)
    weights_str_ = mon.parse_menu_string(weights_)
    weights      = click.prompt(click.style(f"Weights {weights_str_}", fg="bright_green", bold=True), type=str, default=weights or "")
    weights      = weights if weights not in [None, ""] else None
    if weights is not None:
        if isinstance(weights, str):
            weights = mon.to_list(weights)
        weights = [weights_[int(w)] if mon.is_int(w) else w for w in weights]
        weights = [w.replace("'", "") for w in weights]
    # Predict data
    if mode in ["predict", "online", "instance"]:
        data_     = mon.list_datasets(project_root=root, task=task, mode="predict")
        data_str_ = mon.parse_menu_string(data_)
        data      = data.replace(",", ",\n    ") if isinstance(data, str) else data
        data	  = click.prompt(click.style(f"Predict(s) {data_str_}", fg="bright_green", bold=True), type=str, default=data)
        data 	  = mon.to_list(data)
        data 	  = [data_[int(d)] if mon.is_int(d) else d for d in data]
    # Fullname
    fullname    = mon.Path(config).stem if config not in [None, "None", ""] else model_name
    fullname    = click.prompt(click.style(f"Save name: {fullname}", fg="bright_green", bold=True), type=str, default=fullname)
    # Device
    devices_    = mon.list_devices()
    devices_str = mon.parse_menu_string(devices_)
    device      = "auto" if model_name in mon.list_mon_models(mode=mode, task=task) and mode == "train" else device
    device      = click.prompt(click.style(f"Device {devices_str}", fg="bright_green", bold=True), type=str, default=device or "cuda:0")
    device	    = devices_[int(device)] if mon.is_int(device) else device
    # Training Flags
    if mode in ["train", "online", "instance"]:  # Epochs
        epochs = click.prompt(click.style(f"Epochs              ", fg="bright_yellow", bold=True), type=int, default=epochs)
        epochs = None if epochs < 0 else epochs
        steps  = click.prompt(click.style(f"Steps               ", fg="bright_yellow", bold=True), type=int, default=steps)
        steps  = None if steps  < 0 else steps
    # Predict Flags
    if mode in ["predict", "online", "instance"]:  # Image size
        imgsz_       = imgsz
        imgsz        = click.prompt(click.style(f"Image size          ", fg="bright_yellow", bold=True), type=str, default=imgsz)
        imgsz        = mon.to_int_list(imgsz)
        imgsz        = imgsz[0] if len(imgsz) == 1 else imgsz
        imgsz        = None if imgsz < 0 else imgsz
        resize       = "yes" if imgsz_ not in [None, -1] else "no"
        resize       = click.prompt(click.style(f"Resize?     [yes/no]", fg="bright_yellow", bold=True), type=str, default=resize)
        resize       = True if resize       == "yes" else False
        benchmark    = click.prompt(click.style(f"Benchmark?  [yes/no]", fg="bright_yellow", bold=True), type=str, default="yes")
        benchmark    = True if benchmark    == "yes" else False
        save_image   = click.prompt(click.style(f"Save image? [yes/no]", fg="bright_yellow", bold=True), type=str, default="yes")
        save_image   = True if save_image   == "yes" else False
        save_debug   = click.prompt(click.style(f"Save debug? [yes/no]", fg="bright_yellow", bold=True), type=str, default="yes")
        save_debug   = True if save_debug   == "yes" else False
        use_data_dir = click.prompt(click.style(f"Data dir?   [yes/no]", fg="bright_yellow", bold=True), type=str, default="no")
        use_data_dir = True if use_data_dir == "yes" else False
    # Common Flags
    # Exist OK?
    exist_ok = click.prompt(click.style(f"Exist OK?   [yes/no]", fg="bright_yellow", bold=True), type=str, default=exist_ok)
    exist_ok = True if exist_ok == "yes" else False
    # Use Verbose
    verbose  = click.prompt(click.style(f"Verbosity?  [yes/no]", fg="bright_yellow", bold=True), type=str, default=verbose)
    verbose  = True if verbose  == "yes" else False
    
    print("\n")
    
    # Run
    if mode in ["install"]:
        args = {
            "root" : root,
            "model": model,
        }
        run_install(args)
    elif mode in ["train"]:
        args = {
            "task"    : task,
            "mode"    : mode,
            "config"  : config,
            "arch"    : arch,
            "model"   : model,
            "root"    : root,
            "project" : project,
            "variant" : variant,
            "fullname": fullname,
            "save_dir": save_dir,
            "weights" : weights,
            "device"  : device,
            "epochs"  : epochs,
            "steps"   : steps,
            "exist_ok": exist_ok,
            "verbose" : verbose,
        }
        run_train(args=args)
    elif mode in ["predict"]:
        args = {
            "task"        : task,
            "mode"        : mode,
            "config"      : config,
            "arch"        : arch,
            "model"       : model,
            "data"        : data,
            "root"        : root,
            "project"     : project,
            "variant"     : variant,
            "fullname"    : fullname,
            "save_dir"    : save_dir,
            "weights"     : weights,
            "device"      : device,
            "imgsz"       : imgsz,
            "resize" 	  : resize,
            "benchmark"   : benchmark,
            "save_image"  : save_image,
            "save_debug"  : save_debug,
            "use_data_dir": use_data_dir,
            "verbose"     : verbose,
        }
        run_predict(args=args)
    elif mode in ["online", "instance"]:
        args = {
            "task"        : task,
            "mode"        : mode,
            "config"      : config,
            "arch"        : arch,
            "model"       : model,
            "data"        : data,
            "root"        : root,
            "project"     : project,
            "variant"     : variant,
            "fullname"    : fullname,
            "save_dir"    : save_dir,
            "weights"     : weights,
            "device"      : device,
            "epochs"      : epochs,
            "steps"       : steps,
            "imgsz"       : imgsz,
            "resize" 	  : resize,
            "benchmark"   : benchmark,
            "save_image"  : save_image,
            "save_debug"  : save_debug,
            "use_data_dir": use_data_dir,
            "verbose"     : verbose,
        }
        run_online(args=args)
    else:
        raise ValueError(
            f":param:`mode` must be one of ``'train'``, ``'predict'``, "
            f"``'online'``, ``'instance'``, ``'metric'``, or ``'plot'``, "
            f"but got {mode}."
        )
        

if __name__ == "__main__":
    main()

# endregion
