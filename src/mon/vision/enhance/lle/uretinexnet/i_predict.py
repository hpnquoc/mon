#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Implements the paper: "URetinex-Net: Retinex-based Deep Unfolding Network for
Low-light-Image-Enhancement," CVPR 2022.

References:
    - https://github.com/AndersonYong/URetinex-Net
"""

import time

import box
import torchvision.transforms as transforms

import mon
from mon import nn
from network.Math_Module import P, Q
from network.decom import Decom
from utils import *

current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# ----- Utils -----
def benchmark(model: torch.nn.Module):
    flops, params = mon.compute_efficiency_score(model=model)
    mon.console.log(f"FLOPs : {flops:.4f}")
    mon.console.log(f"Params: {params:.4f}")


def one2three(x):
    return torch.cat([x, x, x], dim=1).to(x)


class Inference(nn.Module):
    
    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        # Loading decomposition model
        self.model_Decom_low = Decom()
        self.model_Decom_low = load_initialize(self.model_Decom_low, self.opts["decom_model_low_weights"])
        # Loading R; old_model_opts; and L model
        self.unfolding_opts, self.model_R, self.model_L = load_unfolding(self.opts["unfolding_model_weights"])
        # Loading adjustment model
        self.adjust_model    = load_adjustment(self.opts["adjust_model_weights"])
        self.P = P()
        self.Q = Q()
        transform = [
            transforms.ToTensor(),
            # transforms.Resize(1280),
        ]
        self.transform = transforms.Compose(transform)
        # mon.console.log(self.model_Decom_low)
        # mon.console.log(self.model_R)
        # mon.console.log(self.model_L)
        # mon.console.log(self.adjust_model)
        # time.sleep(8)

    def unfolding(self, input_low_img):
        for t in range(self.unfolding_opts.round):      
            if t == 0:  # Initialize R0, L0
                P, Q = self.model_Decom_low(input_low_img)
            else:  # Update P and Q
                w_p = (self.unfolding_opts.gamma + self.unfolding_opts.Roffset * t)
                w_q = (self.unfolding_opts.lamda + self.unfolding_opts.Loffset * t)
                P   = self.P(I=input_low_img, Q=Q, R=R, gamma=w_p)
                Q   = self.Q(I=input_low_img, P=P, L=L, lamda=w_q)
            R = self.model_R(r=P, l=Q)
            L = self.model_L(l=Q)
        return R, L
    
    def illumination_adjust(self, L, ratio):
        ratio = torch.ones(L.shape).cuda() * ratio
        return self.adjust_model(l=L, alpha=ratio)
    
    def forward(self, input_low_img):
        if torch.cuda.is_available():
            input_low_img = input_low_img.cuda()
        with torch.no_grad():
            start_time = time.time()
            R, L       = self.unfolding(input_low_img)
            High_L     = self.illumination_adjust(L, self.opts["ratio"])
            I_enhance  = High_L * R
            run_time   = (time.time() - start_time)
        return I_enhance, run_time

    def run(self, low_img_path):
        low_img           = self.transform(Image.open(str(low_img_path)).convert("RGB")).unsqueeze(0)
        enhance, run_time = self.forward(input_low_img=low_img)
        """
        file_name = os.path.basename(self.opts.img_path)
        name      = file_name.split('.')[0]
        if not os.path.exists(self.opts.output):
            os.makedirs(self.opts.output)
        save_path = os.path.join(self.opts.output, file_name.replace(name, "%s_%d_URetinexNet"%(name, self.opts.ratio)))
        np_save_TensorImg(enhance, save_path)
        mon.console.log("================================= time for %s: %f============================"%(file_name, p_time))
        """
        return enhance, run_time
        

# ----- Predict -----
@torch.no_grad()
def predict(args: dict | box.Box) -> str:
    # Start
    mon.print_run_summary(args)

    # Device
    device = mon.set_device(args.device)

    # Seed
    mon.set_random_seed(args.seed)

    # Data I/O
    data_name, data_loader = mon.parse_data_loader(args.data, args.root, False, verbose=False)

    # Pretrained
    args.decom_model_low_weights = mon.ZOO_DIR / args.decom_model_low_weights
    args.unfolding_model_weights = mon.ZOO_DIR / args.unfolding_model_weights
    args.adjust_model_weights    = mon.ZOO_DIR / args.adjust_model_weights

    # Model
    model = Inference(args)
    model = model.to(device)
    model.eval()
    
    # Benchmark
    if args.benchmark:
        benchmark(model)
    
    # Predict
    timers = mon.TimeProfiler()
    with mon.create_progress_bar() as pbar:
        for i, datapoint in pbar.track(
            sequence    = enumerate(data_loader),
            total       = len(data_loader),
            description = f"[bright_yellow]Predicting"
        ):
            # Preprocess
            timers.preprocess.tick()
            path   = mon.Path(datapoint["meta"]["path"])
            timers.preprocess.tock()

            # Infer
            timers.infer.tick()
            outputs = model.run(path)
            timers.infer.tock()

            # Postprocess
            timers.postprocess.tick()
            enhanced, _ = outputs
            timers.postprocess.tock()

            # Save
            if args.save_image:
                out_dir  = mon.parse_output_dir(args.save_dir, data_name, mon.SAVE_IMAGE_DIR, path, args.keep_subdirs, args.save_nearby)
                out_path = out_dir / f"{path.stem}{mon.SAVE_IMAGE_EXT}"
                mon.save_image(enhanced, out_path)
        
    # Finish
    timers.print()
    return str(args.save_dir)


# ----- Main -----
def main() -> str:
    args = mon.parse_predict_args(model_root=current_dir)
    predict(args)


if __name__ == "__main__":
    main()
