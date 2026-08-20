"""Validated probability ensembling utilities."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def weighted_probability_average(
    predictions: Sequence[np.ndarray],
    weights: Sequence[float] | None = None,
) -> np.ndarray:
    """Average same-shaped probability matrices with explicit normalized weights."""
    if not predictions:
        raise ValueError("At least one prediction matrix is required")
    arrays = [np.asarray(values, dtype=float) for values in predictions]
    if any(values.shape != arrays[0].shape for values in arrays):
        raise ValueError("Ensemble prediction shapes differ")
    if any(not np.isfinite(values).all() for values in arrays):
        raise ValueError("Ensemble predictions contain non-finite values")
    if any(((values < 0) | (values > 1)).any() for values in arrays):
        raise ValueError("Ensemble predictions must lie in [0, 1]")
    raw_weights = np.ones(len(arrays), dtype=float) if weights is None else np.asarray(weights)
    if raw_weights.shape != (len(arrays),) or not np.isfinite(raw_weights).all():
        raise ValueError("Ensemble weights are invalid")
    if (raw_weights < 0).any() or raw_weights.sum() <= 0:
        raise ValueError("Ensemble weights must be non-negative with positive sum")
    normalized = raw_weights / raw_weights.sum()
    return np.average(np.stack(arrays), axis=0, weights=normalized)


def percentile_ranks(predictions: np.ndarray) -> np.ndarray:
    """Convert each target column to average percentile ranks in `(0, 1]`."""
    values = np.asarray(predictions, dtype=float)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Predictions must be a finite two-dimensional matrix")
    ranks = np.empty(values.shape, dtype=float)
    for column in range(values.shape[1]):
        column_values = values[:, column]
        order = np.argsort(column_values, kind="stable")
        sorted_values = column_values[order]
        sorted_ranks = np.empty(len(order), dtype=float)
        start = 0
        while start < len(order):
            stop = start + 1
            while stop < len(order) and sorted_values[stop] == sorted_values[start]:
                stop += 1
            sorted_ranks[start:stop] = ((start + 1) + stop) / 2.0
            start = stop
        ranks[order, column] = sorted_ranks / len(order)
    return ranks


def weighted_rank_average(
    predictions: Sequence[np.ndarray],
    weights: Sequence[float] | None = None,
) -> np.ndarray:
    """Rank-normalize each model per target, then apply a weighted average."""
    ranked = [percentile_ranks(values) for values in predictions]
    return weighted_probability_average(ranked, weights)
