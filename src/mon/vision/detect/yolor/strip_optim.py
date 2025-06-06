#!/usr/bin/env python
# -*- coding: utf-8 -*-

from mon import core
from utils.general import (
    strip_optimizer,
)

current_file = core.Path(__file__).absolute()
current_dir  = current_file.parents[0]


# ----- Main -----
def main():
    for f in [
        "run/train/yolor_d6_aicity_2024_fisheye8k_1536_epoch_100/weights/best_p.pt",
    ]:
        f = core.Path(f)
        if f.exists():
            strip_optimizer(f)  # strip optimizers


if __name__ == "__main__":
    main()
