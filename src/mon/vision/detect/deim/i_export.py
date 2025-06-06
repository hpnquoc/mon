"""
D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright (c) 2023 lyuwenyu. All Rights Reserved.
"""

import os
import sys

import box

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import torch
from engine.core import YAMLConfig
import mon

current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# ----- Export -----
class Model(torch.nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.model         = cfg.model.deploy()
        self.postprocessor = cfg.postprocessor.deploy()

    def forward(self, images, orig_target_sizes):
        outputs = self.model(images)
        outputs = self.postprocessor(outputs, orig_target_sizes)
        return outputs


@torch.no_grad()
def export(args: dict | box.Box) -> str:
    # Start
    mon.print_run_summary(args)

    # Device
    device = mon.set_device(args.device)

    # Seed
    mon.set_random_seed(args.seed)

    # Pretrained
    pretrained = args.resume
    if args.weights and args.weights.is_weights_file(exist=True):
        pretrained = args.weights
    if pretrained and pretrained.is_weights_file(exist=True):
        mon.console.log(f"Pretrained: {pretrained}.")
    else:
        raise ValueError(f"Invalid weights file: {pretrained}.")

    # Model
    cfg_path     = current_dir / "option" / args.cfg
    updated_cfg  = args.updated_cfg
    updated_cfg |= {"resume": str(pretrained)} if pretrained else {}
    updated_cfg |= {
        "device": device,
        "seed"  : args.seed,
    }
    cfg = YAMLConfig(cfg_path=str(cfg_path), root=str(args.root), **updated_cfg)

    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    if pretrained:
        checkpoint = torch.load(pretrained, map_location="cpu")
        if "ema" in checkpoint:
            state = checkpoint["ema"]["module"]
        else:
            state = checkpoint["model"]
    else:
        raise AttributeError("Only support resume to load model.state_dict by now.")

    # Load train mode state and convert to deploy mode
    cfg.model.load_state_dict(state)
    model = Model(cfg)

    # Export ONNX model
    imgsz = args.imgsz[0] if isinstance(args.imgsz, list | tuple) else args.imgsz
    data  = torch.rand(32, 3, imgsz, imgsz)
    size  = torch.tensor([[imgsz, imgsz]])
    _     = model(data, size)
    dynamic_axes = {
        "images"           : {0: "N"},
        "orig_target_sizes": {0: "N"}
    }
    output_file  = pretrained.parent / f"{pretrained.stem}.onnx"
    
    torch.onnx.export(
        model,
        (data, size),
        output_file,
        input_names         = ["images", "orig_target_sizes"],
        output_names        = ["labels", "boxes", "scores"],
        dynamic_axes        = dynamic_axes,
        opset_version       = 16,
        verbose             = False,
        do_constant_folding = True,
    )

    check = True
    if check:
        import onnx
        onnx_model = onnx.load(output_file)
        onnx.checker.check_model(onnx_model)
        print("Check export onnx model done...")

    if args["simplify"]:
        import onnx
        import onnxsim
        dynamic = True
        # input_shapes = {'images': [1, 3, 640, 640], 'orig_target_sizes': [1, 2]} if dynamic else None
        input_shapes = {"images": data.shape, "orig_target_sizes": size.shape} if dynamic else None
        onnx_model_simplify, check = onnxsim.simplify(output_file, test_input_shapes=input_shapes)
        onnx.save(onnx_model_simplify, output_file)
        print(f"Simplify onnx model {check}...")


# ----- Main -----
def main() -> str:
    args = mon.parse_predict_args(model_root=current_dir)
    export(args)


if __name__ == "__main__":
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--config", "-c", type=str, default="options/deim_dfine/dfine_hgnetv2_l_coco80.yml")
    # parser.add_argument("--resume", "-r", type=str)
    # parser.add_argument("--check",        action="store_true", default=True)
    # parser.add_argument("--simplify",     action="store_true", default=True)
    # args = parser.parse_args()
    # main(args)
    main()
