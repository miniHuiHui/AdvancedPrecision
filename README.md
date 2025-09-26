# AdvancedPrecision

This repository sketches a practical path for bringing emulated FP256 (and
optionally native FP128) support to PyTorch by combining floating-point
expansions with Tensor Core accelerated kernels.

## 1. Reality check & targets

* **Native Tensor Core support today** – Tensor Cores natively accelerate
  FP64/FP32/TF32/FP16/BF16/FP8. Hopper and Blackwell introduce FP64 Tensor
  Cores, but nothing beyond that precision is currently available in
  hardware.
* **True FP128 exposure** – CUDA (≥ 12.8 on supported platforms) now exposes
  quad precision `__float128` math on device. The library should dispatch to
  it when present, otherwise fall back to software emulation.
* **FP256 and higher** – must be emulated via software formats such as
  floating-point expansions or super-accumulators layered on top of fast
  GEMMs.

## 2. Numeric strategy

1. **FP128 (binary128)** – Prefer native device support when available and
   expose it as an optional dtype in the API.
2. **Floating-Point Expansions (FPE)** – Represent each high-precision value
   as a non-overlapping sum of several FP64 limbs. Implement arithmetic with
   error-free transforms (TwoSum/TwoProd) and renormalization. 5–6 limbs
   provide roughly FP256 precision.
3. **Accurate reductions** – Use long-accumulator approaches (ExBLAS) or
   Ozaki-style splitting to ensure deterministic, highly accurate dot
   products and reductions.

## 3. Tensor Core accelerated FP256

* Decompose each operand into `k` FP64 limbs (`A = ΣA[i]`, `B = ΣB[j]`).
* Compute pairwise products `C[i,j] = A[i] @ B[j]` using CUTLASS FP64
  Tensor-Core kernels (fall back to FP32/FP16/FP8 variants as needed).
* Accumulate the partial results with super-accumulators or FPE-aware
  renormalization to recover `k` limbs in the output.
* Reduce the k² multiplication cost with Karatsuba or Toom–Cook for
  larger `k`.

For dot products, reductions, Softmax, and LayerNorm, apply Ozaki splitting
so low-precision Tensor-Core math combines into high-precision results.

## 4. PyTorch integration

* Store limbs in a structure-of-arrays layout (`TensorList[limb]`) for
  coalesced access.
* Provide a Python wrapper (`torch_fpN.Tensor`) that tracks limbs and routes
  to C++/CUDA extension kernels registered with ATen.
* Implement autograd by reusing the same FPE kernels for backward passes.
* Allow runtime selection of limb count (`k`) for precision/performance
  trade-offs and mixed-precision interoperability.

## 5. Kernel stack & building blocks

* **GEMM/Conv:** CUTLASS/CuTe FP64 (or lower precision) Tensor-Core kernels.
* **Reductions:** ExBLAS-style accumulators with deterministic NCCL support.
* **Elementwise:** FPE arithmetic (add/mul/div/sqrt) using fused multiply-add
  heavy kernels inspired by CAMPARY.
* **Ozaki backend:** optional path that leverages FP8/FP16 Tensor Cores for
  bandwidth-bound workloads on older GPUs.

## 6. API sketch

```python
class FPx(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *limbs):
        return _fpx_ops.matmul(limbsA=limbs[:k], limbsB=limbs[k:2*k], k=k)

    @staticmethod
    def backward(ctx, *grad_limbs):
        # reuse the same kernels with transposed operands
        ...
```

The underlying C++/CUDA extension should:

1. Launch Tensor-Core GEMMs for every limb pair.
2. Accumulate partial results with a superaccumulator.
3. Renormalize back into `k` output limbs and return them to PyTorch.

## 7. Performance considerations

* Favor structure-of-arrays layouts to keep memory access coalesced.
* Renormalize within each tile to reduce global memory traffic.
* Manage register pressure carefully; keep `k` modest and fuse accumulation
  with MMA epilogues.
* Target FP256 with `k = 5–6`, validating accuracy against MPFR references.

## 8. Reference Python implementation

A pure-Python prototype that exercises the floating-point expansion arithmetic
described above lives in the ``advanced_precision`` package.  It implements
error-free transforms (TwoSum/TwoProd), renormalization, vector dot products,
and matrix multiplication using a limb-based representation.  While this
prototype executes on the CPU, it provides a convenient harness for validating
numerical behavior, writing unit tests, and experimenting with limb counts
before porting the kernels to CUDA.

Example usage:

```python
from advanced_precision import from_float, add, negate, fpe_dot

# Construct double-double (k=2) numbers
a = from_float(1e16, k=2)
b = from_float(1.0, k=2)

# (1e16 + 1) - 1e16 retains the low limb instead of rounding away
diff = add(add(a, b, k=2), negate(a), k=2)
print(sum(diff))  # ~1.0

# High-accuracy dot product with heavy cancellation
vec_a = [from_float(v, k=4) for v in (1e16, 1.0, -1e16)]
vec_b = [from_float(1.0, k=4) for _ in range(3)]
print(sum(fpe_dot(vec_a, vec_b, k=4)))  # ~1.0
```

Run ``pytest`` to execute the accompanying accuracy checks that compare the
expansion routines against Python ``decimal`` references.

### Simple NumPy neural network helper

The ``advanced_precision.simple_nn`` module provides a small utility for
experimenting with NumPy-based feed-forward networks without pulling in a full
deep-learning stack.  Supply weight matrices, bias vectors, and an input batch,
then call ``forward`` to obtain the activations of the final layer:

```python
import numpy as np

from advanced_precision.simple_nn import forward

weights = [
    np.array([[0.2, -0.4], [0.7, 0.1]]),
    np.array([[0.5], [-0.3]]),
]
biases = [
    np.array([0.1, -0.2]),
    np.array([0.05]),
]
inputs = np.array([[1.0, 0.5]])

output = forward(weights, biases, inputs)
```

### PyTorch integration helpers

The ``advanced_precision.torch_support`` module exposes a light-weight bridge
for experimenting with floating-point expansions inside PyTorch.  When PyTorch
is installed, importing ``advanced_precision`` also re-exports the helpers so
you can write:

```python
import torch
from advanced_precision import (
    TorchExpansionTensor,
    set_default_torch_limb_count,
    torch_to_tensor,
)

set_default_torch_limb_count(6)  # ≈ FP256

weights = torch.randn(128, 128)
expansion_weights = TorchExpansionTensor.from_tensor(weights)

# High-precision matmul using limb schoolbook products + renormalization
activations = TorchExpansionTensor.from_tensor(torch.randn(32, 128))
expanded_output = expansion_weights.matmul(activations)
float_output = expanded_output.to_tensor()
```

The helpers operate purely on ``torch.float64`` tensors so they work on CPU or
GPU, and they preserve PyTorch autograd by never leaving the tensor world.  The
global ``set_default_torch_limb_count`` knob allows you to switch between
double-double, quad-double, or ~FP256 precision without modifying the rest of
your PyTorch model code.

## 9. Validation & roadmap

* Compare against CPU references (MPFR/Boost.Multiprecision) for correctness.
* Add property tests covering cancellation and renormalization stability.
* Roadmap: (v0) double-double; (v1) quad-double with deterministic
  all-reduce; (v2) generic `k` (≈ FP256) and Ozaki backend; (v3) automatic
  native FP128 dispatch where available.
