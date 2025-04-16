#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements main running pipeline."""

import subprocess

import mon
import menu_rich

current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# ----- Train -----
def run_train(args: dict):
    # Get user input
    task         = args["task"]
    mode         = args["mode"]
    config       = args["config"]
    root         = mon.Path(args["root"])
    arch         = args["arch"]
    model        = args["model"]
    data         = args["data"]
    fullname     = args["fullname"]
    save_dir     = args["save_dir"]
    weights      = args["weights"]
    device       = args["device"]
    imgsz        = args["imgsz"]
    resize       = args["resize"]
    epochs       = args["epochs"]
    steps        = args["steps"]
    benchmark    = args["benchmark"]
    save_image   = args["save_image"]
    save_debug   = args["save_debug"]
    # use_fullname = args["use_fullname"]
    keep_subdirs = args["keep_subdirs"]
    exist_ok     = args["exist_ok"]
    verbose      = args["verbose"]
    
    assert root.exists()
    
    # Parse arguments
    use_extra_model = mon.is_extra_model(model)
    model_root      = mon.parse_model_dir(arch, model)
    model           = mon.parse_model_name(model)
    fullname        = fullname if fullname not in [None, "None", ""] else config.stem
    config          = mon.parse_config_file(
        config       = config,
        project_root = root,
        model_root   = model_root,
        weights_path = weights,
    )
    assert config not in [None, "None", ""]
    # save_dir = save_dir or mon.parse_save_dir(root/"run"/"train", arch, model, data, project, variant)
    weights  = mon.to_str(weights, ",")
    
    kwargs   = {
        "--config"  : config,
        "--root"    : str(root),
        "--arch"    : arch,
        "--model"   : model,
        "--fullname": fullname,
        "--save-dir": str(save_dir),
        "--weights" : weights,
        "--device"  : device,
        # "--imgsz"   : imgsz,
        "--epochs"  : epochs,
        "--steps"   : steps,
    }
    flags  = ["--benchmark"]    if benchmark    else []
    flags += ["--save-image"]   if save_image   else []
    flags += ["--save-debug"]   if save_debug   else []
    flags += ["--keep-subdirs"] if keep_subdirs else []
    flags += ["--exist-ok"]     if exist_ok     else []
    flags += ["--verbose"]      if verbose      else []
    
    # Parse script file
    if use_extra_model:
        # torch_distributed_launch = mon.EXTRA_MODELS[arch][model]["torch_distributed_launch"]
        script_file = mon.EXTRA_MODELS[arch][model]["model_dir"] / "i_train.py"
        python_call = ["python"]
        # device      = mon.parse_device(device)
        # if isinstance(device, list) and torch_distributed_launch:
        #     python_call = [
        #         f"python",
        #         f"-m",
        #         f"torch.distributed.launch",
        #         f"--nproc_per_node={str(len(device))}",
        #         f"--master_port=9527"
        #     ]
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
    

# ----- Predict -----
def run_predict(args: dict):
    # Get user input
    task         = args["task"]
    mode         = args["mode"]
    config       = args["config"]
    root         = mon.Path(args["root"])
    arch         = args["arch"]
    model        = args["model"]
    data         = args["data"]
    fullname     = args["fullname"]
    save_dir     = args["save_dir"]
    weights      = args["weights"]
    device       = args["device"]
    imgsz        = args["imgsz"]
    resize       = args["resize"]
    epochs       = args["epochs"]
    steps        = args["steps"]
    benchmark    = args["benchmark"]
    save_image   = args["save_image"]
    save_debug   = args["save_debug"]
    use_fullname = args["use_fullname"]
    keep_subdirs = args["keep_subdirs"]
    exist_ok     = args["exist_ok"]
    verbose      = args["verbose"]
    
    assert root.exists()
    
    # Parse arguments
    use_extra_model = mon.is_extra_model(model)
    model_root      = mon.parse_model_dir(arch, model)
    model           = mon.parse_model_name(model)
    data            = mon.to_list(data)
    fullname        = fullname if fullname not in [None, "None", ""] else model
    config          = mon.parse_config_file(
        project_root = root,
        model_root   = model_root,
        weights_path = weights,
        config       = config,
    )
    # assert config not in [None, "None", ""]
    config  = config or ""
    weights = mon.to_str(weights, ",")
    
    for d in data:
        kwargs  = {
            "--config"  : config,
            "--root"    : str(root),
            "--arch"    : arch,
            "--model"   : model,
            "--data"    : d,
            "--fullname": fullname,
            "--save-dir": str(save_dir),
            "--weights" : weights,
            "--device"  : device,
            "--imgsz"   : imgsz,
            # "--epochs"  : epochs,
            # "--steps"   : steps,
        }
        flags   = ["--resize"]       if resize       else []
        flags  += ["--benchmark"]    if benchmark    else []
        flags  += ["--save-image"]   if save_image   else []
        flags  += ["--save-debug"]   if save_debug   else []
        flags  += ["--keep-subdirs"] if keep_subdirs else []
        flags  += ["--exist-ok"]     if exist_ok     else []
        flags  += ["--verbose"]      if verbose      else []
        
        # Parse script file
        if use_extra_model:
            script_file = mon.EXTRA_MODELS[arch][model]["model_dir"] / "i_predict.py"
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
        

# ----- Main -----
def main():
    defaults = vars(mon.parse_default_args(name="main"))
    menu     = menu_rich.RunmlCLI(defaults=defaults)
    args     = menu.prompt_args()
    
    # Run
    if args["mode"] in ["train"]:
        run_train(args=args)
    elif args["mode"] in ["predict"]:
        run_predict(args=args)
    else:
        raise ValueError(f"Unknown mode: {args['mode']}.")
        

if __name__ == "__main__":
    main()

# endregion
