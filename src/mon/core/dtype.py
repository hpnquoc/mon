#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Generic Data Types.

This module implements data handling capabilities, including lists,
dictionaries, tuples, sets, and more advanced data structures from the
`collections` module.
"""

from __future__ import annotations

__all__ = [
    "Enum",
    "concat_lists",
    "flatten_models_dict",
    "get_module_vars",
    "intersect_dicts",
    "intersect_ordered_dicts",
    "is_float",
    "is_int",
    "iter_to_iter",
    "iter_to_list",
    "iter_to_tuple",
    "shuffle_dict",
    "split_list",
    "to_1list",
    "to_1tuple",
    "to_2list",
    "to_2tuple",
    "to_3list",
    "to_3tuple",
    "to_4list",
    "to_4tuple",
    "to_5list",
    "to_5tuple",
    "to_6list",
    "to_6tuple",
    "to_float",
    "to_float_list",
    "to_int",
    "to_int_list",
    "to_list",
    "to_nlist",
    "to_ntuple",
    "to_pair",
    "to_quadruple",
    "to_single",
    "to_str",
    "to_triple",
    "to_tuple",
    "unique",
    "upcast",
]

import enum
import itertools
import random
import re
from collections import OrderedDict
from types import ModuleType
from typing import Any, Callable, Iterable

import numpy as np
import torch


# region Enum

class Enum(enum.Enum):
    """An extension of Python enum.Enum."""
    
    @classmethod
    def random(cls):
        """Return a random enum.
    
        Returns:
            Enum: A random enum member from the class.
        """
        return random.choice(list(cls))

    @classmethod
    def random_value(cls):
        """Return a random enum value.
    
        Returns:
            Any: A random value from the enum.
        """
        return cls.random().value

    @classmethod
    def keys(cls) -> list:
        """Return a list of all enums.
    
        Returns:
            list: A list of all enum members.
        """
        return list(cls)

    @classmethod
    def values(cls) -> list:
        """Return a list of all enums' values.
    
        Returns:
            list: A list of all enum values.
        """
        return [e.value for e in cls]

# endregion


# region Collection

def intersect_dicts(x: dict, y: dict, exclude: list = []) -> dict:
    """Find the intersection between two dicts.

    Args:
        x: The first dict.
        y: The second dict.
        exclude: A list of excluding keys. Default: ``[]``.

    Returns:
        A dict that contains only the keys that are in both ``x`` and ``y``, and
        whose values are equal.
    """
    return {k: v for k, v in x.items() if k in y and k not in exclude and v == y[k]}


def intersect_ordered_dicts(x: OrderedDict, y: OrderedDict, exclude: list = []) -> OrderedDict:
    """Find the intersection between two OrderedDicts.

    Args:
        x: The first ordered dict.
        y: The second ordered dict.
        exclude: A list of excluding keys. Default: ``[]``.

    Returns:
        An ``OrderedDict`` that contains only the keys that are in both ``x``
        and ``y``, and whose values are equal.
    """
    return OrderedDict((k, v) for k, v in x.items() if k in y and k not in exclude and v == y[k])


def shuffle_dict(x: dict) -> dict:
    """Shuffle a ``dict`` randomly.

    Args:
        x: The dictionary to shuffle.

    Returns:
        A new dictionary with the keys shuffled.
    """
    keys = list(x.keys())
    random.shuffle(keys)
    return {key: x[key] for key in keys}


def flatten_models_dict(x: dict) -> dict:
    """Flatten a nested dictionary of models into a single dictionary.

    Args:
        x: A nested dictionary of models.

    Returns:
        A flattened dictionary where each key is from the nested dictionaries,
        and each value is either the original value or a dictionary with an
        added ``"arch"`` key.
    """
    return {k2: {**v2, "arch": k1} if isinstance(v2, dict) else v2
            for k1, v1 in x.items() for k2, v2 in v1.items()}

# endregion


# region Module

def get_module_vars(module: ModuleType) -> dict:
    """Return all public variables of a module in a ``dict``.

    Args:
        module: The module from which to retrieve public variables.

    Returns:
        A dictionary containing the public variables of the module.
    """
    return {
        k: v for k, v in vars(module).items()
        if not (
            k == "__init__"
            or callable(k)
            or isinstance(v, ModuleType)
            or k.startswith(("_", "__", "annotations"))
        )
    }

# endregion


# region Numeric

def is_int(x: Any) -> bool:
    """Check if a value can be converted to an integer.

    Args:
        x: The value to check.

    Returns:
        ``True`` if the value can be converted to an integer, ``False`` otherwise.
    """
    try:
        int(x)
        return True
    except ValueError:
        return False
    
    
def is_float(x: Any) -> bool:
    """Check if a value can be converted to a float.

    Args:
        x: The value to check.

    Returns:
        ``True`` if the value can be converted to a float, ``False`` otherwise.
    """
    try:
        float(x)
        return True
    except ValueError:
        return False


def to_int(x: Any) -> int | None:
    """Convert a value to an integer.

    Args:
        x: The value to convert.

    Returns:
        The converted integer, or ``None`` if the input is ``None``.

    Raises:
        ValueError: If the input is a string that cannot be converted to an integer.
    """
    if x is None:
        return None
    if isinstance(x, str) and not is_int(x):
        raise ValueError(f"`x` must be a digit string, but got {x} ({type(x)}).")
    return int(x)


def to_float(x: Any) -> float | None:
    """Convert a value to a float.

    Args:
        x: The value to convert.

    Returns:
        The converted float, or ``None`` if the input is ``None``.

    Raises:
        ValueError: If the input is a string that cannot be converted to a float.
    """
    if x is None:
        return None
    if isinstance(x, str) and not is_float(x):
        raise ValueError(f"`x` must be a digit string, but got {x} ({type(x)}).")
    return float(x)

# endregion


# region Parsing

def upcast(x: torch.Tensor | np.ndarray, keep_type: bool = False) -> torch.Tensor | np.ndarray:
    """Protect from numerical overflows in multiplications by upcasting to the
    equivalent higher type.

    Args:
        x: An input of type ``numpy.ndarray`` or ``torch.Tensor``.
        keep_type: If ``True``, keep the same type (int32  -> int64). Else
            upcast to a higher type (int32 -> float32).
            
    Returns:
        A variable of higher type.
    """
    if x.dtype in {torch.float16, np.float16}:
        return x.to(torch.float32) if isinstance(x, torch.Tensor) else x.astype(np.float32)
    elif x.dtype in {torch.float32, np.float32}:
        return x
    elif x.dtype in {torch.int8, np.int16}:
        return x.to(torch.int16) if keep_type else x.to(torch.float16) if isinstance(x, torch.Tensor) else x.astype(np.float32)
    elif x.dtype in {torch.int16, np.int32}:
        return x.to(torch.int32) if keep_type else x.to(torch.float32) if isinstance(x, torch.Tensor) else x.astype(np.float32)
    elif x.dtype == torch.int32:
        return x
    return x

# endregion


# region Sequence

def concat_lists(x: list[list]) -> list:
    """Concatenate a list of lists into a flattened list.

    Args:
        x: A list of lists to concatenate.

    Returns:
        A single flattened list containing all elements from the input lists.
    """
    return list(itertools.chain.from_iterable(x))


def iter_to_iter(x: Iterable, item_type: type, return_type: type = None):
    """Convert an ``Iterable`` object to a desired sequence type specified
    by the ``return_type``. Also, cast each item into the desired ``item_type``.

    Args:
        x: An ``Iterable`` object.
        item_type: The item type.
        return_type: The desired iterable type. Default: ``None``.

    Returns:
        An ``Iterable`` object cast to the desired type.
    """
    if not isinstance(x, (list, tuple, dict)):
        raise TypeError(f"`x` must be a `list`, `tuple`, or `dict`, but got {type(x)}.")
    x = map(item_type, x)
    return list(x) if return_type is list else tuple(x) if return_type is tuple else x


def iter_to_list(x: Iterable, item_type: type) -> list:
    """Convert an arbitrary ``Iterable`` object to a ``list``.

    Args:
        x: An ``Iterable`` object to convert.
        item_type: The type to which each item in the iterable should be cast.

    Returns:
        A list containing the items from the iterable, cast to the specified type.
    """
    return list(map(item_type, x))


def iter_to_tuple(x: Iterable, item_type: type) -> tuple:
    """Convert an arbitrary ``Iterable`` object to a ``tuple``.

    Args:
        x: An ``Iterable`` object to convert.
        item_type: The type to which each item in the iterable should be cast.

    Returns:
        A tuple containing the items from the iterable, cast to the specified type.
    """
    return tuple(map(item_type, x))


def split_list(x: list, n: int | list[int]) -> list[list]:
    """Slice a single `list` into a list of lists.

    Args:
        x: A ``list`` object.
        n: A number of sub-lists, or a ``list`` of integers to specify the
            length of each sub-list.
        
    Returns:
        A ``list`` of lists.
    
    Examples:
        >>> x = [1, 2, 3, 4, 5, 6]
        >>> y = split_list(x, n=2)          # [1, 2, 3], [4, 5, 6]
        >>> z = split_list(x, n=[1, 3, 2])  # [1], [2, 3, 4], [5, 6]
    """
    if isinstance(n, int):
        if len(x) % n != 0:
            raise ValueError(f"`x` cannot be evenly split into {n} sub-lists, "
                             f"length of `x` is {len(x)}.")
        n = [n] * (len(x) // n)

    if sum(n) != len(x):
        raise ValueError(f"The total length of new sub-lists must match the "
                         f"length of `x`, but got {sum(n)} != {len(x)}.")

    y = [x[idx: idx + size] for idx, size in zip(range(0, len(x), n[0]), n)]
    return y


def to_list(x: Any, sep: list[str] = [",", ";", ":"]) -> list:
    """Convert an arbitrary value into a ``list``.

    Args:
        x: An arbitrary value.
        sep: A ``list`` of delimiters to split a string.

    Returns:
        A list representation of the input value.
    """
    if isinstance(x, list):
        return x
    if isinstance(x, (tuple, dict)):
        return list(x) if isinstance(x, tuple) else list(x.values())
    if isinstance(x, str):
        x = re.sub(r"^\s+|\s+$", "", x)
        x = re.sub(r"\s", "", x)
        for s in sep:
            if s in x:
                return x.split(s)
        return [x]
    return [x] if x is not None else []


def to_int_list(x: Any, sep: list[str] = [",", ";", ":"]) -> list[int]:
    """Convert a string into a ``list`` of ``int``.

    Args:
        x: The input value to convert, which can be of any type.
        sep: A list of delimiters to split the string. Default: [",", ";", ":"].

    Returns:
        A list of integers obtained by splitting the input string.
    """
    return [int(i) for i in to_list(x, sep=sep)]


def to_float_list(x: Any, sep: list[str] = [",", ";", ":"]) -> list[float]:
    """Convert a string into a `list` of `float`.

    Args:
        x: The input value to convert, which can be of any type.
        sep: A list of delimiters to split the string. Default: [",", ";", ":"].

    Returns:
        A list of floats obtained by splitting the input string.
    """
    return [float(i) for i in to_list(x, sep=sep)]


def to_nlist(n: int) -> Callable[[Any], list]:
    """Return a function that converts an input to a list of length ``n``.

    Args:
        n: The desired length of the list.

    Returns:
        A function that takes an input and converts it to a list of length ``n``.
    """
    def parse(x) -> list:
        x = list(x) if isinstance(x, Iterable) else [x]
        return x * n if len(x) == 1 else x
    return parse


to_1list = to_nlist(1)
to_2list = to_nlist(2)
to_3list = to_nlist(3)
to_4list = to_nlist(4)
to_5list = to_nlist(5)
to_6list = to_nlist(6)


def to_tuple(x: Any) -> tuple:
    """Convert an arbitrary value into a tuple.

    Args:
        x: The value to convert, which can be of any type.

    Returns:
        A tuple representation of the input value.
    """
    if isinstance(x, list):
        return tuple(x)
    if isinstance(x, dict):
        return tuple(x.values())
    return tuple(x)


def to_ntuple(n: int) -> Callable[[Any], tuple]:
    """Take an integer ``n`` and return a function that takes an ``Iterable``
    object and returns a `tuple` of length ``n``.

    Args:
        n: The number of elements in the ``tuple``.

    Returns:
        A function that takes an input and returns a ``tuple`` of that input
        repeated ``n`` times.
    """
    def parse(x) -> tuple:
        if isinstance(x, Iterable):
            x = tuple(x)
            if len(x) == 1:
                x = tuple(itertools.repeat(x[0], n))
        else:
            x = tuple(itertools.repeat(x, n))
        return x
    
    return parse


to_1tuple    = to_ntuple(1)
to_2tuple    = to_ntuple(2)
to_3tuple    = to_ntuple(3)
to_4tuple    = to_ntuple(4)
to_5tuple    = to_ntuple(5)
to_6tuple    = to_ntuple(6)
to_single    = to_ntuple(1)
to_pair      = to_ntuple(2)
to_triple    = to_ntuple(3)
to_quadruple = to_ntuple(4)


def unique(x: list | tuple) -> list | tuple:
    """Get unique items from a ``list`` or ``tuple``.

    Args:
        x: A ``list`` or ``tuple`` from which to get unique items.

    Returns:
        A ``list`` or ``tuple`` containing unique items from the input.
    """
    return type(x)(set(x))

# endregion


# region String

def to_str(x: Any, sep: str = ",") -> str:
    """Convert an arbitrary value into a string, with elements separated by a delimiter.

    Args:
        x: The value to convert, which can be of any type.
        sep: The delimiter to use for separating elements. Default: ",".

    Returns:
        A string representation of the input value, with elements separated by
        the delimiter.
    """
    if isinstance(x, dict):
        x = x.values()
    if not isinstance(x, (list, tuple)):
        x = [x]
    x = [str(xi) for xi in x]
    return sep.join(x) if len(x) > 1 else x[0]
    
# endregion
