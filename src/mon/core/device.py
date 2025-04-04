#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Handles device management and memory usage."""

__all__ = [
    "get_cuda_memory_usages",
    "get_memory_usages",
    "get_model_device",
    "is_rank_zero",
    "list_cuda_devices",
    "list_devices",
    "parse_device",
    "pynvml_available",
    "set_device",
]

import os
from typing import Any

import psutil
import torch

from mon.constants import MemoryUnit

try:
    import pynvml
    pynvml_available = True
except ImportError:
    pynvml_available = False


# ----- Retrieve -----
def list_cuda_devices() -> str | None:
    """Lists all available CUDA devices on the machine.

    Returns:
        String of CUDA devices (e.g., ``cuda:0,1,2``) or ``None`` if none.
    """
    if torch.cuda.is_available():
        num_devices = torch.cuda.device_count()
        cuda_str    = "cuda:" + ",".join(str(i) for i in range(num_devices))
        return cuda_str
    return None


def list_devices() -> list[str]:
    """Lists all available devices on the machine.

    Returns:
        List of device strings including ``auto``, ``cpu``, and CUDA if available.
    """
    devices = ["auto", "cpu"]
    if torch.cuda.is_available():
        num_devices  = torch.cuda.device_count()
        devices.extend(f"cuda:{i}" for i in range(num_devices))
        all_cuda_str = "cuda:" + ",".join(str(i) for i in range(num_devices))
        if all_cuda_str != "cuda:0":
            devices.append(all_cuda_str)
    return devices


def get_cuda_memory_usages(device: int = 0, unit: MemoryUnit = MemoryUnit.GB) -> list[int]:
    """Gets GPU memory status as a list of total, used, and free memory.

    Args:
        device: GPU device index. Default is ``0``.
        unit: Memory unit (e.g., ``GB``). Default is ``MemoryUnit.GB``.

    Returns:
        List of [total, used, free] memory values in specified unit.
    """
    pynvml.nvmlInit()
    unit  = MemoryUnit.from_value(unit)
    info  = pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(device))
    ratio = MemoryUnit.name_to_byte()[unit]
    return [
        info.total / ratio,  # total
        info.used  / ratio,  # used
        info.free  / ratio   # free
    ]


def get_memory_usages(unit: MemoryUnit = MemoryUnit.GB) -> list[int]:
    """Gets RAM status as a list of total, used, and free memory.

    Args:
        unit: Memory unit (e.g., ``GB``). Default is ``MemoryUnit.GB``.

    Returns:
        List of [total, used, free] memory values in specified unit.
    """
    memory = psutil.virtual_memory()
    ratio  = MemoryUnit.name_to_byte()[MemoryUnit.from_value(unit)]
    return [
        memory.total     / ratio,  # total
        memory.used      / ratio,  # used
        memory.available / ratio   # free
    ]


def get_model_device(model: torch.nn.Module) -> torch.device:
    """Gets the device of a model's parameters.

    Args:
        model: Model to check.

    Returns:
        ``torch.device`` where model parameters reside.
    """
    return next(model.parameters()).device


# ----- Update -----
def set_device(device: Any, use_single_device: bool = True) -> torch.device:
    """Sets the device for the current process.

    Args:
        device: Device to set (e.g., CUDA index, list, or string).
        use_single_device: If ``True``, uses first device from list. Default is ``True``.

    Returns:
        Selected ``torch.device``, defaults to ``cpu`` if CUDA unavailable.
    """
    device = parse_device(device)
    if isinstance(device, list) and use_single_device:
        device = device[0]
    return torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")


# ----- Convert -----
def parse_device(device: Any) -> list[int] | int | str:
    """Parses a device spec into a list, integer, or string.

    Args:
        device: Device to parse (e.g., ``torch.device``, int, str, or ``None``).

    Returns:
        List of ints for multi-device, int for single, or str (``cpu`` or ``mps``).
    """
    if isinstance(device, torch.device):
        return device
    
    if not device or device in ["", "cpu"]:
        return "cpu"
    if device in ["mps", "mps:0"]:
        return device
    if isinstance(device, int):
        return [device]
    if isinstance(device, str):
        device = (device.lower()
                  .replace("cuda:", "")
                  .replace("none", "")
                  .translate(str.maketrans("", "", "()[ ]' ")))
        return [int(x) for x in device.split(",")] \
            if "," in device \
            else [0] if not device else device
    return device


# ----- Validation Check -----
def is_rank_zero() -> bool:
    """Checks if current process is rank zero in distributed training.

    Notes:
        Based on PyTorch Lightning's DDP documentation, "LOCAL_RANK" and "NODE_RANK"
        environment variables indicate child processes for GPUs. Absence of both
        denotes the main process (rank zero).

    Returns:
        ``True`` if current process is rank zero, ``False`` otherwise.
    """
    return "LOCAL_RANK" not in os.environ and "NODE_RANK" not in os.environ
