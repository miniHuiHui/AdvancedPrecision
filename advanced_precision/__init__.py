"""Lightweight floating-point expansion utilities.

This package provides pure-Python reference implementations of the core
building blocks described in the repository README.  The goal is to supply a
CPU-only model that mirrors the algorithms which will later be ported to
CUDA/Tensor-Core kernels.
"""

from .fpe import (
    two_sum,
    fast_two_sum,
    two_prod,
    renormalize,
    add,
    mul,
    negate,
    scale,
    from_float,
    to_float,
)
from .linalg import fpe_dot, fpe_matmul, from_float_matrix, to_float_matrix

__all__ = [
    "two_sum",
    "fast_two_sum",
    "two_prod",
    "renormalize",
    "add",
    "mul",
    "negate",
    "scale",
    "from_float",
    "to_float",
    "fpe_dot",
    "fpe_matmul",
    "from_float_matrix",
    "to_float_matrix",
]
