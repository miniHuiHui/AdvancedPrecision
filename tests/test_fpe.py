from decimal import Decimal, getcontext

import pytest

from advanced_precision import (
    add,
    fpe_dot,
    fpe_matmul,
    from_float,
    from_float_matrix,
    negate,
    to_float,
    to_float_matrix,
)


def test_double_double_cancellation():
    k = 2
    big = from_float(1e16, k)
    small = from_float(1.0, k)

    summed = add(big, small, k)
    diff = add(summed, negate(big), k)

    assert to_float(diff) == pytest.approx(1.0, rel=0, abs=1e-12)


def test_dot_product_high_precision():
    k = 4
    vec_a = [from_float(v, k) for v in (1e16, 1.0, -1e16)]
    vec_b = [from_float(1.0, k) for _ in range(3)]

    result = fpe_dot(vec_a, vec_b, k)
    assert sum(result) == pytest.approx(1.0, rel=0, abs=1e-9)


def test_matrix_multiplication_matches_decimal():
    getcontext().prec = 80
    k = 4

    matrix_a = [[1e16, 1.0], [1.0, -1e16]]
    matrix_b = [[1.0, 2.0], [3.0, 4.0]]

    a_exp = from_float_matrix(matrix_a, k)
    b_exp = from_float_matrix(matrix_b, k)
    c_exp = fpe_matmul(a_exp, b_exp, k)
    c_float = to_float_matrix(c_exp)

    expected = []
    for row in range(len(matrix_a)):
        expected_row = []
        for col in range(len(matrix_b[0])):
            total = Decimal(0)
            for idx in range(len(matrix_b)):
                total += Decimal(str(matrix_a[row][idx])) * Decimal(
                    str(matrix_b[idx][col])
                )
            expected_row.append(total)
        expected.append(expected_row)

    for i, row in enumerate(c_float):
        for j, value in enumerate(row):
            assert value == pytest.approx(float(expected[i][j]), rel=0, abs=1e-8)
