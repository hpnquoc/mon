#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from cldm.hack import disable_verbosity

disable_verbosity()

import random
import argparse
import copy

import torch.optim
import torch.utils
import cv2
import einops
import numpy as np
import torch
from pytorch_lightning import seed_everything

import mon
from annotator.util import HWC3, resize_image
from cldm.model import create_model, load_state_dict
from ldm.models.diffusion.dpm_solver import DPMSolverSampler

console      = mon.console
current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# region Predict

def calculate_model_parameters(model):
    return sum(p.numel() for p in model.parameters())


def process(
    model,
    diffusion_sampler,
    input_image,
    prompt          : str   = "",
    num_samples     : int   = 1,
    image_resolution: int   = 512,
    diffusion_steps : int   = 10,
    guess_mode      : bool  = False,
    strength        : float = 1.0,
    scale           : float = 9.0,
    seed            : int   = 0,
    eta             : float = 0.0,
    use_float16     : bool  = True,
):
    with torch.no_grad():
        detected_map = resize_image(HWC3(input_image), image_resolution)
        H, W, C      = detected_map.shape
        
        if use_float16:
            control = torch.from_numpy(detected_map.copy()).cuda().to(dtype=torch.float16) / 255.0
        else:
            control = torch.from_numpy(detected_map.copy()).cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, "b h w c -> b c h w").clone()
        ae_hs   = model.encode_first_stage(control * 2 - 1)[1]
        
        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)
        
        # if args.save_memory:
        #     model.low_vram_shift(is_diffusing=False)
        
        cond    = {
            "c_concat"   : [control],
            "c_crossattn": [model.get_unconditional_conditioning(num_samples)]
        }
        un_cond = {
            "c_concat"   : None if guess_mode else [control],
            "c_crossattn": [model.get_unconditional_conditioning(num_samples)]
        }
        shape   = (4, H // 8, W // 8)
        
        # if args.save_memory:
        #     model.low_vram_shift(is_diffusing=True)
        
        model.control_scales   = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, intermediates = diffusion_sampler.sample(
            diffusion_steps, num_samples, shape, cond,
            verbose                      = False,
            eta                          = eta,
            unconditional_guidance_scale = scale,
            unconditional_conditioning   = un_cond,
            dmp_order                    = 3,
        )
        
        # if args.save_memory:
        #     model.low_vram_shift(is_diffusing=False)
        
        if use_float16:
            x_samples = model.decode_new_first_stage(samples.to(dtype=torch.float16), ae_hs)
        else:
            x_samples = model.decode_new_first_stage(samples, ae_hs)
        x_samples = (einops.rearrange(x_samples, "b c h w -> b h w c") * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
        
        results = [x_samples[i] for i in range(num_samples)]
    return results


def predict(args: argparse.Namespace):
    # General config
    data         = args.data
    save_dir     = args.save_dir
    # weights    = args.weights or mon.ZOO_DIR / "vision/enhance/llie/quadprior/quadprior/coco/control_sd15_coco_final.ckpt"
    weights      = mon.ZOO_DIR / args.weights
    device       = mon.set_device(args.device)
    imgsz        = args.imgsz[0]
    resize       = args.resize
    benchmark    = args.benchmark
    save_image   = args.save_image
    save_debug   = args.save_debug
    use_fullpath = args.use_fullpath
    use_float16  = args.use_float16
    
    config_path = current_dir / args.config_path  # "./models/cldm_v15.yaml"
    init_ckpt   = mon.ZOO_DIR / "vision/enhance/llie/quadprior/quadprior/coco/control_sd15_init.ckpt"
    ae_ckpt     = mon.ZOO_DIR / "vision/enhance/llie/quadprior/quadprior/coco/ae_epoch=00_step=7000.ckpt"
    
    # Model
    model          = create_model(config_path=config_path).cpu()
    state_dict     = load_state_dict(str(init_ckpt), location="cpu")
    new_state_dict = {}
    for s in state_dict:
        if "cond_stage_model.transformer" not in s:
            new_state_dict[s] = state_dict[s]
    model.load_state_dict(new_state_dict)
    # Insert new layers in ControlNet (sorry for the ugliness)
    model.add_new_layers()
    # Load trained checkpoint
    state_dict     = load_state_dict(weights, location="cpu")
    new_state_dict = {}
    for sd_name, sd_param in state_dict.items():
        if "_forward_module.control_model" in sd_name:
            new_state_dict[sd_name.replace("_forward_module.control_model.", "")] = sd_param
    model.control_model.load_state_dict(new_state_dict)
    # Load bypass decoder
    model.change_first_stage(ae_ckpt)
    
    if use_float16:
        model = model.to(device).to(dtype=torch.float16)
    else:
        model = model.to(device)
    diffusion_sampler = DPMSolverSampler(model)
    
    if benchmark:
        # total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        flops, params, avg_time = mon.compute_efficiency_score(
            model      = copy.deepcopy(model),
            image_size = imgsz,
            channels   = 3,
            runs       = 1000,
            use_cuda   = True,
            verbose    = False,
        )
        total_params = calculate_model_parameters(model)
        console.log(f"FLOPs        = {flops:.4f}")
        console.log(f"Params       = {params:.4f}")
        console.log(f"Time         = {avg_time:.17f}")
        console.log(f"Total Params = {total_params:.4f}")
    
    # Data I/O
    console.log(f"[bold red]{data}")
    data_name, data_loader, data_writer = mon.parse_io_worker(
        src         = data,
        dst         = save_dir,
        to_tensor   = False,
        denormalize = True,
        verbose     = False,
    )
    
    # Predicting
    timer = mon.Timer()
    with torch.no_grad():
        with mon.get_progress_bar() as pbar:
            for i, datapoint in pbar.track(
                sequence    = enumerate(data_loader),
                total       = len(data_loader),
                description = f"[bright_yellow] Predicting"
            ):
                # Input
                image       = datapoint.get("image")
                meta        = datapoint.get("meta")
                image_path  = mon.Path(meta["path"])
                input_image = image
                h0, w0      = input_image.shape[0], input_image.shape[1]
                
                # Infer
                timer.tick()
                # If you set num_samples > 1, process will return multiple results
                output      = process(
                    model, diffusion_sampler,
                    input_image      = input_image,
                    num_samples      = 1,
                    image_resolution = imgsz,
                    use_float16      = use_float16,
                )[0]
                timer.tock()
                
                # Post-processing
                output = mon.resize(output, (h0, w0), interpolation=cv2.INTER_LINEAR)
                output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
                
                # Save
                if save_image:
                    if use_fullpath:
                        rel_path    = image_path.relative_path(data_name)
                        output_path = save_dir / rel_path.parent / image_path.name
                    else:
                        output_path = save_dir / data_name / image_path.name
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(output_path), output)
        
        avg_time = float(timer.avg_time)
        console.log(f"Average time: {avg_time}")

# endregion


# region Main

def main() -> str:
    args = mon.parse_predict_args(model_root=current_dir)
    predict(args)


if __name__ == "__main__":
    main()
    
# endregion
