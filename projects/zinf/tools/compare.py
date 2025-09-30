#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

import cv2
import numpy as np

import mon

mon.dev()

current_file = mon.Path(__file__).absolute()
root_dir     = current_file.parents[1]
data_dir     = root_dir / "data"
run_dir      = root_dir / "run"

models = [
    # "zinf_siren",
    "zinf_finer",
    "zinf_pe_siren",
    # "zinf_pe_finer",
    # "zinf_siren_d",
]


def compare(data: str) -> str:
    colie_dir  = run_dir  / "predict" / "colie" / "colie" / data / "pred"
    model_dirs = [run_dir / "predict" / "zinf"  / m / data / "pred" for m in models]
    
    colie_files = sorted(list(colie_dir.glob("*")))
    for colie_file in colie_files:
        model_files = [md / colie_file.name for md in model_dirs]
        
        colie_image  = cv2.imread(str(colie_file))
        empty_image  = np.zeros((colie_image.shape[0], colie_image.shape[1], 3), np.uint8)
        concat_image = cv2.vconcat([colie_image, empty_image])
        for model_file in model_files:
            model_image  = cv2.imread(str(model_file))
            diff_image   = cv2.absdiff(model_image, colie_image)
            gray_diff    = cv2.cvtColor(diff_image, cv2.COLOR_BGR2GRAY)
            norm_diff    = cv2.normalize(gray_diff, None, 0, 255, cv2.NORM_MINMAX)
            heatmap      = cv2.applyColorMap(norm_diff, cv2.COLORMAP_JET)
            concat_image = cv2.hconcat([concat_image, cv2.vconcat([model_image, heatmap])])
            
        cv2.imshow("Compare", concat_image)
        cv2.waitKey(0)
        

# ----- Main -----
def main() -> str:
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model", type=str, default="zinf_siren_d")
    parser.add_argument("--data",  type=str, default="dicm")
    args = parser.parse_args()
    # compare(args.model, args.data)
    compare(args.data)


if __name__ == "__main__":
    main()
