#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The MON framework."""

__author__  = "Long H. Pham"
__version__ = "2.8.0"


# Import core packages
import time
__start = time.time()

from mon.core import *
from mon.constants import *
# import mon.datasets  # Register datasets

__end = time.time()
console.log(f"`mon` loaded in: {__end - __start:.4f} seconds.")


# Import development packages
def dev(verbose: bool = False):
    start = time.time()
    
    import mon.datasets  # Register datasets
    import mon.vision    # Register vision models
    
    end = time.time()
    if verbose:
        console.log(f"`mon-dev` loaded in: {end - start:.4f} seconds.")
