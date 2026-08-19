"""Deterministic 2.5D slice sampling utilities."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def uniform_indices(length: int, count: int) -> np.ndarray:
    """Return exactly ``count`` stable indices, repeating edges for short series."""
    if length < 1 or count < 1:
        raise ValueError("length and count must be positive")
    return np.rint(np.linspace(0, length - 1, count)).astype(np.int64)


def random_uniform_indices(length: int, count: int, rng: np.random.Generator) -> np.ndarray:
    """Stratified random sampling with one draw per equal-width slice bin."""
    if length < 1 or count < 1:
        raise ValueError("length and count must be positive")
    edges = np.linspace(0, length, count + 1)
    result = []
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        start = min(int(np.floor(lower)), length - 1)
        stop = min(max(int(np.ceil(upper)), start + 1), length)
        result.append(int(rng.integers(start, stop)))
    return np.asarray(sorted(result), dtype=np.int64)


def neighborhood_indices(
    center: int,
    length: int,
    offsets: Sequence[int] = (-1, 0, 1),
) -> np.ndarray:
    """Clamp neighboring indices at series boundaries for a 2.5D stack."""
    if length < 1:
        raise ValueError("length must be positive")
    if not 0 <= center < length:
        raise ValueError("center is outside the series")
    return np.asarray([min(max(center + int(offset), 0), length - 1) for offset in offsets])
