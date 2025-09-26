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

import importlib.util

_torch_spec = importlib.util.find_spec("torch")
if _torch_spec is not None:
    from .torch_support import (
        TorchExpansionTensor,
        add as torch_add,
        from_tensor as torch_from_tensor,
        get_default_limb_count as get_default_torch_limb_count,
        is_torch_available,
        matmul as torch_matmul,
        mul as torch_mul,
        set_default_limb_count as set_default_torch_limb_count,
        to_tensor as torch_to_tensor,
    )
else:  # pragma: no cover - exercised when torch is missing
    TorchExpansionTensor = None  # type: ignore[assignment]
    torch_add = None  # type: ignore[assignment]
    torch_from_tensor = None  # type: ignore[assignment]
    get_default_torch_limb_count = None  # type: ignore[assignment]

    def is_torch_available() -> bool:  # type: ignore[no-redef]
        return False

    torch_matmul = None  # type: ignore[assignment]
    torch_mul = None  # type: ignore[assignment]
    set_default_torch_limb_count = None  # type: ignore[assignment]
    torch_to_tensor = None  # type: ignore[assignment]

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

if _torch_spec is not None:
    __all__.extend(
        [
            "TorchExpansionTensor",
            "torch_add",
            "torch_from_tensor",
            "get_default_torch_limb_count",
            "is_torch_available",
            "torch_matmul",
            "torch_mul",
            "set_default_torch_limb_count",
            "torch_to_tensor",
        ]
    )
