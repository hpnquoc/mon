#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Encode and decode subset IDs for Pretrained or FishEye8K datasets."""

import argparse
import json
from datetime import datetime

import numpy as np

import mon

current_file = mon.Path(__file__).absolute()
current_dir  = current_file.parents[0]

alphabet     = "0123456789ABCDEFGHJKLMNPRSTUV"
subset_ids   = {
    "coco"       : 1,  # 2^0
    "object365"  : 2,  # 2^1
    "visdrone"   : 2,  # 2^2
    "train"      : 3,  # 2^3
    "train_fixed": 4,  # 2^4
    "val"        : 5,  # 2^5
    "val_fixed"  : 6,  # 2^6
    "test"       : 7,  # 2^7
    "syn"        : 8,  # 2^8
}


def encode_subset(subsets: list[str]) -> str:
    """
    Encodes a list of subsets from Pretrained or FishEye8K into a 2-character base-32 ID.

    Args:
        subsets: List of subset names (e.g., ["coco", "train"]).

    Returns:
        2-character string ID (using 0-9, A-V, excluding I, O, Q, W, X, Y, Z).

    Example:
        >>> encode_subset(["coco", "train"])
        "09"
    """
    subset_bits = {k: 2 ** v for k, v in subset_ids.items()}
    bitmask     = sum(subset_bits[subset] for subset in subsets)
    # Convert to base-32: first char is bitmask // 32, second is bitmask % 32
    first       = bitmask // 32
    second      = bitmask  % 32
    return alphabet[first] + alphabet[second]


def decode_subset(id: str) -> list[str]:
    """
    Decodes a 2-character base-32 ID into the corresponding list of subset names.

    Args:
        id: 2-character string ID (using 0-9, A-V).

    Returns:
        List of subset names corresponding to the ID.

    Example:
        >>> decode_subset("09")
        ["coco", "train"]
    """
    # Convert base-32 to integer: first char * 32 + second char index
    if len(id) != 2 or not all(c in alphabet for c in id):
        return []
    bitmask     = alphabet.index(id[0]) * 32 + alphabet.index(id[1])
    subset_bits = {2 ** v: k for k, v in subset_ids.items()}
    return [name for bit, name in subset_bits.items() if bitmask & bit]


if __name__ == "__main__":
    subsets = [
        # "coco",
        # "object365",
        "visdrone",
        "train",
        # "train_fixed",
        "val",
        # "val_fixed",
        # "test",
        "syn",
    ]
    print(encode_subset(subsets=subsets))
    # print(decode_subset(id="098"))
