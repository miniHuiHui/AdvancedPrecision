"""Linear algebra helpers built on top of floating-point expansions."""
from __future__ import annotations

from typing import List, Sequence

from .fpe import add, mul, renormalize

Expansion = Sequence[float]
Matrix = Sequence[Sequence[Expansion]]


def from_float_matrix(matrix: Sequence[Sequence[float]], k: int) -> List[List[List[float]]]:
    """Convert a real-valued matrix into a matrix of expansions."""

    return [[renormalize([value], k) for value in row] for row in matrix]


def to_float_matrix(matrix: Matrix) -> List[List[float]]:
    """Collapse a matrix of expansions back into plain floating values."""

    return [[float(sum(expansion)) for expansion in row] for row in matrix]


def fpe_dot(vec_a: Sequence[Expansion], vec_b: Sequence[Expansion], k: int) -> List[float]:
    """Compute a high-precision dot product using expansions."""

    if len(vec_a) != len(vec_b):
        raise ValueError("dot product requires equally sized vectors")

    acc: List[float] = [0.0]
    for a, b in zip(vec_a, vec_b):
        prod = mul(a, b, k=None)
        acc = add(acc, prod, k=None)
    return renormalize(acc, k)


def fpe_matmul(a: Matrix, b: Matrix, k: int) -> List[List[List[float]]]:
    """Matrix multiplication using floating-point expansions."""

    if not a or not b:
        return []

    num_rows = len(a)
    num_cols = len(b[0])
    shared = len(b)
    if any(len(row) != shared for row in a):
        raise ValueError("incompatible shapes for matrix multiplication")
    if any(len(row) != num_cols for row in b):
        raise ValueError("matrix b must be rectangular")

    result: List[List[List[float]]] = []
    for i in range(num_rows):
        row: List[List[float]] = []
        for j in range(num_cols):
            col = [b[r][j] for r in range(shared)]
            dot = fpe_dot(a[i], col, k)
            row.append(dot)
        result.append(row)
    return result
