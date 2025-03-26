#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""THOP: PyTorch-OpCounter.

This module extend ``thop.profile``.
"""

from __future__ import annotations

# noinspection PyUnresolvedReferences
from thop import *
from thop.profile import register_hooks
from thop.vision.basic_hooks import *

default_dtype  = torch.float64


def custom_profile(
    model         : nn.Module,
    inputs        : dict,
    custom_ops    : dict = None,
    verbose       : bool = True,
    ret_layer_info: bool = False,
    report_missing: bool = False,
):
    """Extend ``thop.profile`` to work with ``mon.nn.model.Model`` custom forward pass.
    
    Args:
        model: The PyTorch model to profile.
        inputs: The input data to profile.
        custom_ops: A dictionary mapping PyTorch module types to functions that compute
            the number of operations. Default: ``None``.
        verbose: If ``True``, print hook registration messages. Default: ``True``.
        ret_layer_info: If ``True``, return detailed layer info. Default: ``False``.
        report_missing: If ``True``, log missing op rules. Default: ``False``.

    Returns:
        The total number of operations and parameters in the model.
        If ``ret_layer_info`` is ``True``, also returns a dict of per-layer stats:
        ``(total_ops, total_params, layer_info_dict)``.
    """
    handler_collection = {}
    types_collection   = set()
    custom_ops         = custom_ops or {}
    if report_missing:
        verbose = True

    def add_hooks(m: nn.Module):
        """Register hooks to track total operations and parameters."""
        m.register_buffer("total_ops",    torch.zeros(1, dtype=torch.float64))
        m.register_buffer("total_params", torch.zeros(1, dtype=torch.float64))

        m_type = type(m)
        fn = None

        if m_type in custom_ops:
            fn = custom_ops[m_type]
            if m_type not in types_collection and verbose:
                print(f"[INFO] Customize rule [{fn.__qualname__}] for [{m_type}].")
        elif m_type in register_hooks:
            fn = register_hooks[m_type]
            if m_type not in types_collection and verbose:
                print(f"[INFO] Register [{fn.__qualname__}] for [{m_type}].")
        elif m_type not in types_collection and report_missing:
            print(f"[WARN] Cannot find rule for [{m_type}]. Treat as zero MACs and Params.")

        if fn is not None:
            handler_collection[m] = (
                m.register_forward_hook(fn),
                m.register_forward_hook(count_parameters),
            )

        types_collection.add(m_type)

    prev_training_status = model.training
    model.eval()
    model.apply(add_hooks)

    with torch.no_grad():
        model(datapoint=inputs)

    def dfs_count(module: nn.Module, prefix="\t") -> tuple[int, int, dict]:
        """Recursively count operations and parameters."""
        total_ops    = 0
        total_params = 0
        ret_dict     = {}
        for n, m in module.named_children():
            next_dict = {}
            if m in handler_collection and not isinstance(m, (nn.Sequential, nn.ModuleList)):
                m_ops, m_params = m.total_ops.item(), m.total_params.item()
            else:
                m_ops, m_params, next_dict = dfs_count(m, prefix=prefix + "\t")
            ret_dict[n] = (m_ops, m_params, next_dict)
            total_ops += m_ops
            total_params += m_params
        return total_ops, total_params, ret_dict

    total_ops, total_params, ret_dict = dfs_count(model)

    model.train(prev_training_status)
    for m, (op_handler, params_handler) in handler_collection.items():
        op_handler.remove()
        params_handler.remove()
        m._buffers.pop("total_ops")
        m._buffers.pop("total_params")

    return (total_ops, total_params, ret_dict) if ret_layer_info else (total_ops, total_params)
