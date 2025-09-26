"""PyTorch-facing helpers for floating-point expansions."""
from __future__ import annotations

import importlib
from typing import Iterable, List, Sequence

_torch_spec = importlib.util.find_spec("torch")
if _torch_spec is not None:
    torch = importlib.import_module("torch")
else:  # pragma: no cover - exercised only when torch is unavailable
    torch = None  # type: ignore[assignment]

_SPLIT = 134_217_729.0
_DEFAULT_LIMB_COUNT = 6


def is_torch_available() -> bool:
    """Return ``True`` when PyTorch can be imported."""

    return torch is not None


def _require_torch() -> "torch":
    if torch is None:  # pragma: no cover - guarded by availability checks
        raise RuntimeError(
            "PyTorch is not available. Install torch to use the torch integration."
        )
    return torch


def set_default_limb_count(k: int) -> None:
    """Set the default limb count used when wrapping torch tensors."""

    if k <= 0:
        raise ValueError("k must be positive")
    global _DEFAULT_LIMB_COUNT
    _DEFAULT_LIMB_COUNT = k


def get_default_limb_count() -> int:
    """Return the default limb count for new expansions."""

    return _DEFAULT_LIMB_COUNT


def _as_float64(tensor: "torch.Tensor") -> "torch.Tensor":
    torch_lib = _require_torch()
    if tensor.dtype != torch_lib.float64:
        tensor = tensor.to(dtype=torch_lib.float64)
    return tensor


def _split(a: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor"]:
    _require_torch()
    c = _SPLIT * a
    a_big = c - a
    hi = c - a_big
    lo = a - hi
    return hi, lo


def two_sum(a: "torch.Tensor", b: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor"]:
    _require_torch()
    s = a + b
    bp = s - a
    err = (a - (s - bp)) + (b - bp)
    return s, err


def two_prod(a: "torch.Tensor", b: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor"]:
    _require_torch()
    p = a * b
    ah, al = _split(a)
    bh, bl = _split(b)
    err = ((ah * bh - p) + ah * bl + al * bh) + al * bl
    return p, err


def renormalize(
    components: Iterable["torch.Tensor"],
    k: int | None = None,
) -> List["torch.Tensor"]:
    torch_lib = _require_torch()
    components = list(components)
    if not components:
        raise ValueError("renormalize requires at least one component tensor")

    stacked = torch_lib.stack([_as_float64(comp) for comp in components], dim=0)
    magnitudes = torch_lib.abs(stacked)
    _, indices = magnitudes.sort(dim=0, descending=True)
    sorted_components = torch_lib.gather(stacked, 0, indices)

    if k is not None and sorted_components.shape[0] > k:
        kept = sorted_components[: k - 1]
        tail = sorted_components[k - 1 :].sum(dim=0)
        sorted_components = torch_lib.cat(
            [kept, tail.unsqueeze(0)], dim=0
        )
        return renormalize(list(sorted_components), k)

    return [sorted_components[i] for i in range(sorted_components.shape[0])]


def add(
    lhs: Sequence["torch.Tensor"],
    rhs: Sequence["torch.Tensor"],
    k: int | None = None,
) -> List["torch.Tensor"]:
    _require_torch()
    if not lhs or not rhs:
        raise ValueError("expansions must contain at least one limb")
    if k is None:
        k = len(lhs) + len(rhs)
    return renormalize([*_as_float64_list(lhs), *_as_float64_list(rhs)], k=k)


def mul(
    lhs: Sequence["torch.Tensor"],
    rhs: Sequence["torch.Tensor"],
    k: int | None = None,
) -> List["torch.Tensor"]:
    _require_torch()
    if not lhs or not rhs:
        raise ValueError("expansions must contain at least one limb")
    if k is None:
        k = len(lhs) + len(rhs)

    components: List["torch.Tensor"] = []
    for a in _as_float64_list(lhs):
        for b in _as_float64_list(rhs):
            prod, err = two_prod(a, b)
            components.append(prod)
            components.append(err)
    return renormalize(components, k=k)


def from_tensor(tensor: "torch.Tensor", k: int | None = None) -> List["torch.Tensor"]:
    torch_lib = _require_torch()
    if k is None:
        k = _DEFAULT_LIMB_COUNT
    if k <= 0:
        raise ValueError("k must be positive")
    limbs = [_as_float64(tensor)]
    zero = torch_lib.zeros_like(limbs[0])
    for _ in range(k - 1):
        limbs.append(zero.clone())
    return renormalize(limbs, k=k)


def to_tensor(expansion: Sequence["torch.Tensor"]) -> "torch.Tensor":
    _require_torch()
    if not expansion:
        raise ValueError("expansion must contain at least one limb")
    return torch_lib.stack(_as_float64_list(expansion), dim=0).sum(dim=0)


def matmul(
    lhs: Sequence["torch.Tensor"],
    rhs: Sequence["torch.Tensor"],
    k: int | None = None,
) -> List["torch.Tensor"]:
    torch_lib = _require_torch()
    if not lhs or not rhs:
        raise ValueError("expansions must contain at least one limb")
    if k is None:
        k = len(lhs) + len(rhs)

    components: List["torch.Tensor"] = []
    for a in _as_float64_list(lhs):
        for b in _as_float64_list(rhs):
            components.append(torch_lib.matmul(a, b))
    return renormalize(components, k=k)


class TorchExpansionTensor:
    """A PyTorch tensor represented as a floating-point expansion."""

    def __init__(self, limbs: Sequence["torch.Tensor"], *, k: int | None = None):
        _require_torch()
        if not limbs:
            raise ValueError("limbs cannot be empty")
        processed = _as_float64_list(limbs)
        self._limbs = renormalize(processed, k=k or len(processed))

    @property
    def limbs(self) -> List["torch.Tensor"]:
        return list(self._limbs)

    @property
    def shape(self) -> "torch.Size":
        return self._limbs[0].shape

    @property
    def device(self) -> "torch.device":
        return self._limbs[0].device

    @property
    def dtype(self) -> "torch.dtype":
        return self._limbs[0].dtype

    @property
    def limb_count(self) -> int:
        return len(self._limbs)

    @classmethod
    def from_tensor(cls, tensor: "torch.Tensor", k: int | None = None) -> "TorchExpansionTensor":
        return cls(from_tensor(tensor, k=k))

    def to_tensor(self) -> "torch.Tensor":
        return to_tensor(self._limbs)

    def add(self, other: "TorchExpansionTensor", k: int | None = None) -> "TorchExpansionTensor":
        return TorchExpansionTensor(add(self._limbs, other._limbs, k=k))

    def mul(self, other: "TorchExpansionTensor", k: int | None = None) -> "TorchExpansionTensor":
        return TorchExpansionTensor(mul(self._limbs, other._limbs, k=k))

    def matmul(self, other: "TorchExpansionTensor", k: int | None = None) -> "TorchExpansionTensor":
        return TorchExpansionTensor(matmul(self._limbs, other._limbs, k=k))

    def renormalize(self, k: int | None = None) -> "TorchExpansionTensor":
        return TorchExpansionTensor(renormalize(self._limbs, k=k or self.limb_count))

    def clone(self) -> "TorchExpansionTensor":
        return TorchExpansionTensor([limb.clone() for limb in self._limbs])

    def to(self, *args, **kwargs) -> "TorchExpansionTensor":
        _require_torch()
        moved = [limb.to(*args, **kwargs) for limb in self._limbs]
        return TorchExpansionTensor(moved)

    def detach(self) -> "TorchExpansionTensor":
        return TorchExpansionTensor([limb.detach() for limb in self._limbs])


def _as_float64_list(limbs: Sequence["torch.Tensor"]) -> List["torch.Tensor"]:
    return [_as_float64(limb) for limb in limbs]
