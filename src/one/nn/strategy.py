#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Strategies and Accelerators.

Strategy controls the model distribution across training, evaluation, and
prediction to be used by the `Trainer`.

Strategy is a composition of one Accelerator, one Precision Plugin,
a CheckpointIO plugin and other optional plugins such as the ClusterEnvironment.

References:
    https://pytorch-lightning.readthedocs.io/en/stable/extensions/strategy.html
"""

from __future__ import annotations

from pytorch_lightning.accelerators import *
from pytorch_lightning.strategies import *

from one.constants import ACCELERATORS
from one.constants import STRATEGIES


# H1: - Accelerator ------------------------------------------------------------

ACCELERATORS.register(name="cpu",  module=CPUAccelerator)
ACCELERATORS.register(name="cuda", module=CUDAAccelerator)
ACCELERATORS.register(name="gpu",  module=CUDAAccelerator)
ACCELERATORS.register(name="hpu",  module=HPUAccelerator)
ACCELERATORS.register(name="ipu",  module=IPUAccelerator)
ACCELERATORS.register(name="mps",  module=MPSAccelerator)
ACCELERATORS.register(name="tpu",  module=TPUAccelerator)


# H1: - Strategy ---------------------------------------------------------------

STRATEGIES.register(name="bagua",             module=BaguaStrategy,                 desc="Strategy for training using the Bagua library, with advanced distributed training algorithms and system optimizations.")
STRATEGIES.register(name="collaborative",     module=HivemindStrategy,              desc="Strategy for training collaboratively on local machines or unreliable GPUs across the internet.")
STRATEGIES.register(name="colossalai",        module=ColossalAIStrategy,            desc="Colossal-AI provides a collection of parallel components for you. It aims to support you to write your distributed deep learning models just like how you write your model on your laptop.")
STRATEGIES.register(name="fsdp_native",       module=DDPFullyShardedNativeStrategy, desc="Strategy for Fully Sharded Data Parallel provided by PyTorch.")
STRATEGIES.register(name="fsdp",              module=DDPFullyShardedStrategy,       desc="Strategy for Fully Sharded Data Parallel provided by FairScale.")
STRATEGIES.register(name="ddp_sharded",       module=DDPShardedStrategy,            desc="Optimizer and gradient sharded training provided by FairScale.")
STRATEGIES.register(name="ddp_sharded_spawn", module=DDPSpawnShardedStrategy,       desc="Optimizer sharded training provided by FairScale.")
STRATEGIES.register(name="ddp_spawn",         module=DDPSpawnStrategy,              desc="Spawns processes using the torch.multiprocessing.spawn() method and joins processes after training finishes.")
STRATEGIES.register(name="ddp",               module=DDPStrategy,                   desc="Strategy for multi-process single-device training on one or multiple nodes.")
STRATEGIES.register(name="dp",                module=DataParallelStrategy,          desc="Implements data-parallel training in a single process, i.e., the model gets replicated to each device and each gets a split of the data.")
STRATEGIES.register(name="deepspeed",         module=DeepSpeedStrategy,             desc="Provides capabilities to run training using the DeepSpeed library, with training optimizations for large billion parameter models.")
STRATEGIES.register(name="horovod",           module=HorovodStrategy,               desc="Strategy for Horovod distributed training integration.")
STRATEGIES.register(name="hpu_parallel",      module=HPUParallelStrategy,           desc="Strategy for distributed training on multiple HPU devices.")
STRATEGIES.register(name="hpu_single",        module=SingleHPUStrategy,             desc="Strategy for training on a single HPU device.")
STRATEGIES.register(name="ipu_strategy",      module=IPUStrategy,                   desc="Plugin for training on IPU devices.")
STRATEGIES.register(name="tpu_spawn",         module=TPUSpawnStrategy,              desc="Strategy for training on multiple TPU devices using the torch_xla.distributed.xla_multiprocessing.spawn() method.")
STRATEGIES.register(name="single_tpu",        module=SingleTPUStrategy,             desc="Strategy for training on a single TPU device.")
