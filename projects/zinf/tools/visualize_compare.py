#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

import cv2

import mon

mon.dev()

current_file = mon.Path(__file__).absolute()
root_dir     = current_file.parents[1]
data_dir     = root_dir / "data"
run_dir      = root_dir / "run"


def compare(model: str, data: str) -> str:
    colie_dir = run_dir / "predict" / "colie" / "colie" / data / "pred"
    zinf_dir  = run_dir / "predict" / "zinf"  / model   / data / "pred"
    
    colie_files = sorted(list(colie_dir.glob("*")))
    for colie_file in colie_files:
        zinf_file = zinf_dir / colie_file.name
        
        colie_image  = cv2.imread(str(colie_file))
        zinf_image   = cv2.imread(str(zinf_file))
        diff_image   = cv2.absdiff(colie_image, zinf_image)
        concat_image = cv2.hconcat([colie_image, zinf_image, diff_image])
        cv2.imshow("Compare", concat_image)
        cv2.waitKey(0)
        

# ----- Main -----
def main() -> str:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="zinf_pe_siren_d")
    parser.add_argument("--data",  type=str, default="sice")
    args = parser.parse_args()
    compare(args.model, args.data)
    

if __name__ == "__main__":
    main()
