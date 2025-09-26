"""Floating-point expansion primitives.

The routines implemented here follow the classical error-free transforms used
in algorithms such as Dekker multiplication and Shewchuk style expansion
arithmetic.  They operate purely on Python ``float`` values (IEEE-754 double
precision) which makes the module portable and easy to validate.
"""
from __future__ import annotations

from typing import Iterable, List, Sequence

_SPLIT = 134_217_729.0  # 2**27 + 1, suitable for splitting double precision


def two_sum(a: float, b: float) -> tuple[float, float]:
    """Return ``(sum, error)`` so that ``sum + error == a + b`` exactly."""

    s = a + b
    bp = s - a
    err = (a - (s - bp)) + (b - bp)
    return s, err


def fast_two_sum(a: float, b: float) -> tuple[float, float]:
    """Variant of :func:`two_sum` that assumes ``abs(a) >= abs(b)``."""

    s = a + b
    err = b - (s - a)
    return s, err


def _split(a: float) -> tuple[float, float]:
    """Split ``a`` into two half-size numbers whose sum equals ``a``."""

    c = _SPLIT * a
    a_big = c - a
    hi = c - a_big
    lo = a - hi
    return hi, lo


def two_prod(a: float, b: float) -> tuple[float, float]:
    """Return ``(product, error)`` satisfying ``product + error == a * b``."""

    p = a * b
    ah, al = _split(a)
    bh, bl = _split(b)
    err = ((ah * bh - p) + ah * bl + al * bh) + al * bl
    return p, err


def negate(expansion: Sequence[float]) -> List[float]:
    """Return the additive inverse of ``expansion``."""

    return [-x for x in expansion]


def scale(expansion: Sequence[float], factor: float, k: int | None = None) -> List[float]:
    """Scale an expansion by ``factor`` and renormalize to at most ``k`` limbs."""

    scaled = [x * factor for x in expansion]
    return renormalize(scaled, k)


def renormalize(components: Iterable[float], k: int | None = None) -> List[float]:
    """Normalize ``components`` into a non-overlapping expansion.

    Parameters
    ----------
    components:
        Any iterable of float components that should be combined into an
        expansion.
    k:
        If supplied, the resulting expansion is truncated (with round-to-nearest
        semantics) to at most ``k`` limbs.
    """

    filtered = [c for c in components if c != 0.0]
    if not filtered:
        return [0.0] if (k is None or k > 0) else []

    # Shewchuk style compress: accumulate from small to large magnitudes.
    filtered.sort(key=abs)
    q: List[float] = []
    q_hat = 0.0
    for comp in filtered:
        q_hat, err = two_sum(q_hat, comp)
        if err != 0.0:
            q.append(err)
    q.append(q_hat)

    # Remove zeros introduced by cancellation.
    q = [c for c in q if c != 0.0]
    if not q:
        q = [0.0]

    # Enforce canonical ordering (largest magnitude first).
    q.sort(key=abs, reverse=True)

    if k is not None and len(q) > k:
        # Fold the tail back into the last kept limb, then re-normalize to keep
        # the non-overlapping invariant.
        kept = q[: k - 1]
        tail = sum(q[k - 1 :])
        q = kept + [tail]
        q = renormalize(q, k=None)
        if len(q) > k:
            q = q[:k]
    return q


def add(
    lhs: Sequence[float],
    rhs: Sequence[float],
    k: int | None = None,
) -> List[float]:
    """Add two expansions and renormalize the result."""

    if k is None:
        k = len(lhs) + len(rhs)
    return renormalize([*lhs, *rhs], k)


def mul(
    lhs: Sequence[float],
    rhs: Sequence[float],
    k: int | None = None,
) -> List[float]:
    """Multiply two expansions using schoolbook products + renormalization."""

    if not lhs or not rhs:
        return [0.0]
    if k is None:
        k = len(lhs) + len(rhs)

    components: List[float] = []
    for a in lhs:
        for b in rhs:
            prod, err = two_prod(a, b)
            if err != 0.0:
                components.append(err)
            components.append(prod)
    return renormalize(components, k)


def from_float(value: float, k: int) -> List[float]:
    """Create an expansion representing ``value`` using up to ``k`` limbs."""

    if k <= 0:
        return []
    if value == 0.0:
        return [0.0] + [0.0 for _ in range(k - 1)]
    limbs = [value]
    while len(limbs) < k:
        limbs.append(0.0)
    return renormalize(limbs, k)


def to_float(expansion: Sequence[float]) -> float:
    """Collapse an expansion back into a standard double precision value."""

    return float(sum(expansion))
