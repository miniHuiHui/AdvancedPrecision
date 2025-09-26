"""A tiny subset of NumPy required for the simple neural network tests.

This module is *not* a drop-in replacement for NumPy.  It only implements the
functionality exercised by the repository's unit tests and should be replaced by
an actual NumPy installation for real workloads.
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple, Union

Number = Union[int, float]


def _is_sequence(obj: object) -> bool:
    return isinstance(obj, (list, tuple))


def _convert(data: Union["ndarray", Sequence], dtype=float) -> Union[Number, list]:
    if isinstance(data, ndarray):
        return _convert(data.tolist(), dtype)
    if not _is_sequence(data):
        return dtype(data)
    return [_convert(item, dtype) for item in data]


def _infer_shape(data: Union[Number, list]) -> Tuple[int, ...]:
    if not _is_sequence(data):
        return ()
    if len(data) == 0:
        return (0,)
    inner_shape = _infer_shape(data[0])
    return (len(data),) + inner_shape


def _flatten(data: Union[Number, list]):
    if _is_sequence(data):
        for item in data:
            yield from _flatten(item)
    else:
        yield data


def _apply_unary(data: Union[Number, list], func):
    if _is_sequence(data):
        return [_apply_unary(item, func) for item in data]
    return func(data)


class ndarray:
    """Very small ndarray replacement supporting 1D/2D operations."""

    def __init__(self, data: Union[Sequence, "ndarray"], dtype=float):
        converted = _convert(data, dtype)
        self._data = converted
        self._shape = _infer_shape(self._data)
        self.ndim = len(self._shape)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape

    def tolist(self):
        if isinstance(self._data, list):
            return [item for item in self._data]
        return self._data

    def flatten(self):
        return list(_flatten(self._data))

    def _binary_op(self, other, op):
        if isinstance(other, ndarray):
            if self.shape != other.shape:
                if self.ndim == 2 and other.ndim == 1 and self.shape[1] == other.shape[0]:
                    other_vals = other.flatten()
                    data = [
                        [op(row[j], other_vals[j]) for j in range(self.shape[1])]
                        for row in self._data
                    ]
                    return ndarray(data)
                raise ValueError("shape mismatch for operation")
            if self.ndim == 1:
                data = [op(x, y) for x, y in zip(self._data, other._data)]
            else:
                data = [
                    [op(x, y) for x, y in zip(row_a, row_b)]
                    for row_a, row_b in zip(self._data, other._data)
                ]
            return ndarray(data)
        else:
            if self.ndim == 1:
                data = [op(x, other) for x in self._data]
            else:
                data = [[op(x, other) for x in row] for row in self._data]
            return ndarray(data)

    def __add__(self, other):
        return self._binary_op(other, lambda x, y: x + y)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self._binary_op(other, lambda x, y: x - y)

    def __rsub__(self, other):
        if isinstance(other, ndarray):
            return other.__sub__(self)
        return (-self)._binary_op(other, lambda x, y: x + y)

    def __neg__(self):
        return ndarray(_apply_unary(self._data, lambda x: -x))

    def __matmul__(self, other):
        if not isinstance(other, ndarray):
            other = array(other)
        if self.ndim != 2 or other.ndim != 2:
            raise ValueError("matmul currently supports 2D @ 2D")
        if self.shape[1] != other.shape[0]:
            raise ValueError("shapes are not aligned for matrix multiplication")
        result = []
        other_t = list(zip(*other._data))
        for row in self._data:
            result_row = []
            for col in other_t:
                result_row.append(sum(x * y for x, y in zip(row, col)))
            result.append(result_row)
        return ndarray(result)

    def __array_priority__(self):
        return 1000

    def __iter__(self):
        if self.ndim == 1:
            return iter(self._data)
        return (array(row) for row in self._data)

    def __abs__(self):
        return ndarray(_apply_unary(self._data, abs))

    def max(self):
        return max(self.flatten())

    def __repr__(self):
        return f"ndarray({self._data})"


def array(data: Union[Sequence, ndarray], dtype=float) -> ndarray:
    return ndarray(data, dtype=dtype)


def tanh(values: Union[ndarray, Sequence, Number]) -> ndarray:
    arr = values if isinstance(values, ndarray) else array(values)
    return ndarray(_apply_unary(arr._data, math.tanh))


def maximum(values: Union[ndarray, Sequence, Number], other: Number) -> ndarray:
    arr = values if isinstance(values, ndarray) else array(values)
    return ndarray(_apply_unary(arr._data, lambda x: x if x > other else other))


__all__ = ["array", "ndarray", "tanh", "maximum"]


def isscalar(obj: object) -> bool:
    return not isinstance(obj, (ndarray, list, tuple))


__all__.append("isscalar")


class bool_(int):
    pass


__all__.append("bool_")
