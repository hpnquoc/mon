#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
"""

from __future__ import annotations

import gc
import time

import torch
from flopco import FlopCo

from one.core import console
from one.core import get_gpu_memory
from one.core import Ints
from one.core import MemoryUnit
from one.vision.enhancement.zero_adce import ZeroADCE
from one.vision.enhancement.zero_dce import ZeroDCEVanilla
from one.vision.enhancement.zero_dce_tiny import ZeroDCETiny
from one.vision.enhancement.zero_dcepp import ZeroDCEPPVanilla


# H1: - Functions --------------------------------------------------------------

def measure_ops(model, input_shape: Ints):
    return FlopCo(model, input_shape)


def measure_speed(model, input_shape: Ints, repeat: int = 100, half: bool = True):
    if torch.cuda.is_available():
        input = torch.rand(input_shape).cuda()
        model = model.eval().cuda().cuda()
    else:
        input = torch.rand(input_shape)
        model = model.eval()
        
    if half:
        input = input.half()
        model = model.half()
     
    times  = []
    memory = []
    for e in range(repeat):
        with torch.inference_mode():
            # Start timer
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_max_memory_allocated()
            torch.cuda.synchronize()
            start = time.time()
            # Code to measure
            model(input)
            torch.cuda.synchronize()
            # End timer
            end = time.time()
            times.append(end - start)
        
        if torch.cuda.is_available():
            total, used, free = get_gpu_memory(unit=MemoryUnit.MB)
            memory.append(used)
            
    avg_time = sum(times)  / repeat
    avg_mem  = sum(memory) / repeat
    return avg_time, avg_mem


# H1: - Main -------------------------------------------------------------------

if __name__ == "__main__":
    shape  = (1, 3, 900, 1200)
    models = [
        ZeroADCE(cfg="zeroadce-a", name="ZeroADCE-A"),
        ZeroDCEVanilla(),
        ZeroDCEPPVanilla(),
        ZeroDCETiny(),
        ZeroADCE(cfg="zeroadce-a",       name="ZeroADCE-A"),
        ZeroADCE(cfg="zeroadce-b",       name="ZeroADCE-B"),
        ZeroADCE(cfg="zeroadce-c",       name="ZeroADCE-C"),
        ZeroADCE(cfg="zeroadce-d",       name="ZeroADCE-D"),
        ZeroADCE(cfg="zeroadce-e",       name="ZeroADCE-E"),
        ZeroADCE(cfg="zeroadce-a-large", name="ZeroADCE-A-Large"),
        ZeroADCE(cfg="zeroadce-b-large", name="ZeroADCE-B-Large"),
        ZeroADCE(cfg="zeroadce-c-large", name="ZeroADCE-C-Large"),
        ZeroADCE(cfg="zeroadce-d-large", name="ZeroADCE-D-Large"),
        ZeroADCE(cfg="zeroadce-e-large", name="ZeroADCE-E-Large"),
        ZeroADCE(cfg="zeroadce-a-tiny",  name="ZeroADCE-A-Tiny"),
        ZeroADCE(cfg="zeroadce-b-tiny",  name="ZeroADCE-B-Tiny"),
        ZeroADCE(cfg="zeroadce-c-tiny",  name="ZeroADCE-C-Tiny"),
        ZeroADCE(cfg="zeroadce-d-tiny",  name="ZeroADCE-D-Tiny"),
        ZeroADCE(cfg="zeroadce-e-tiny",  name="ZeroADCE-E-Tiny"),
        ZeroADCE(cfg="zeroadce-abs1",    name="ZeroADCE-ABS1"),
        ZeroADCE(cfg="zeroadce-abs2",    name="ZeroADCE-ABS2"),
        ZeroADCE(cfg="zeroadce-abs3",    name="ZeroADCE-ABS3"),
        ZeroADCE(cfg="zeroadce-abs4",    name="ZeroADCE-ABS4"),
        ZeroADCE(cfg="zeroadce-abs5",    name="ZeroADCE-ABS5"),
        ZeroADCE(cfg="zeroadce-abs6",    name="ZeroADCE-ABS6"),
        ZeroADCE(cfg="zeroadce-abs7",    name="ZeroADCE-ABS7"),
        ZeroADCE(cfg="zeroadce-abs8",    name="ZeroADCE-ABS8"),
        ZeroADCE(cfg="zeroadce-abs9",    name="ZeroADCE-ABS9"),
        ZeroADCE(cfg="zeroadce-abs10",   name="ZeroADCE-ABS10"),
        ZeroADCE(cfg="zeroadce-abs11",   name="ZeroADCE-ABS11"),
        ZeroADCE(cfg="zeroadce-abs12",   name="ZeroADCE-ABS12"),
        ZeroADCE(cfg="zeroadce-abs13",   name="ZeroADCE-ABS13"),
        # ZeroADCEDebug(),
        # ZeroADCETinyDebug(),
    ]
    
    console.log(
        f"{'Model':<20} "
        f"{'MACs (G)':>20} "
        f"{'FLOPs (G)':>20} "
        f"{'Params':>20} "
        f"{'Avg Time (s)':>20} "
        f"{'Memory (MB)':>20} "
    )
    for m in models:
        stats         =   measure_ops(model=m, input_shape=shape)
        speed, memory = measure_speed(model=m, input_shape=shape, repeat=1)
        console.log(
            f"{m.name if hasattr(m, 'name') else m.__class__.__name__:<20} "
            f"{(stats.total_macs / 1000000000):>20.9f} "
            f"{(stats.total_flops / 1000000000):>20.9f} "
            f"{stats.total_params:>20.9f} "
            f"{speed:>20.9f} "
            f"{memory:>20.9f} "
        )
