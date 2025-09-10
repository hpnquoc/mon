#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

import box
import cv2
import torch

import mon

current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# ----- Predict -----
@torch.no_grad()
def predict(args: dict | box.Box) -> str:
    # Parse input arguments
    video_file = current_dir / mon.Path(args.video)
    data_dir   = video_file.parent / video_file.stem
    image_dir  = data_dir  / "image"
    cap        = cv2.VideoCapture(str(video_file))
    
    index = 0
    while True:
        ret, frame = cap.read()
        # Break the loop if reading a frame fails (end of video or error)
        if not ret:
            break
        
        index += 1
        if index % args.skip != 0:
            continue
        
        image_path = image_dir / f"{video_file.stem}_{index:04d}.jpg"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(image_path), frame)
        

# ----- Main -----
def main() -> str:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="data/ic_11_01.mp4", help="Path to image folder")
    parser.add_argument("--skip",  type=int, default=30)
    args = parser.parse_args()
    predict(args)


if __name__ == "__main__":
    main()
