import importlib.util

import pytest

_torch_spec = importlib.util.find_spec("torch")
if _torch_spec is None:  # pragma: no cover - executed when torch missing
    pytest.skip("torch is not installed", allow_module_level=True)

import torch

from advanced_precision import (
    TorchExpansionTensor,
    get_default_torch_limb_count,
    is_torch_available,
    set_default_torch_limb_count,
    torch_add,
    torch_from_tensor,
    torch_matmul,
    torch_mul,
    torch_to_tensor,
)


def test_availability_flag():
    assert is_torch_available() is True


def test_roundtrip_matches_input():
    tensor = torch.randn(3, 4, dtype=torch.float64)
    expansion = torch_from_tensor(tensor, k=5)
    recovered = torch_to_tensor(expansion)
    assert torch.allclose(recovered, tensor, atol=0.0, rtol=0.0)


def test_add_and_mul_preserve_shape():
    a = torch_from_tensor(torch.randn(2, 3), k=4)
    b = torch_from_tensor(torch.randn(2, 3), k=4)
    added = torch_add(a, b, k=6)
    multiplied = torch_mul(a, b, k=6)
    assert all(limb.shape == (2, 3) for limb in added)
    assert all(limb.shape == (2, 3) for limb in multiplied)


def test_matmul_matches_float_result():
    lhs = torch_from_tensor(torch.randn(2, 5), k=5)
    rhs = torch_from_tensor(torch.randn(5, 3), k=5)
    product = torch_matmul(lhs, rhs, k=6)
    recovered = torch_to_tensor(product)
    reference = torch.matmul(torch_to_tensor(lhs), torch_to_tensor(rhs))
    assert torch.allclose(recovered, reference, atol=1e-12, rtol=1e-12)


def test_tensor_wrapper_tracks_precision():
    set_default_torch_limb_count(6)
    assert get_default_torch_limb_count() == 6
    tensor = torch.randn(4, 4)
    wrapped = TorchExpansionTensor.from_tensor(tensor)
    assert wrapped.limb_count == 6
    doubled = wrapped.add(wrapped)
    collapsed = doubled.to_tensor()
    assert torch.allclose(collapsed, tensor.to(torch.float64) * 2.0)
