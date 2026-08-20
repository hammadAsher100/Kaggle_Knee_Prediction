"""Tests for constrained probability averaging."""

import numpy as np
import pytest

from src.inference.ensemble import (
    percentile_ranks,
    weighted_probability_average,
    weighted_rank_average,
)


def test_weighted_probability_average_normalizes_weights() -> None:
    first = np.asarray([[0.2, 0.8]])
    second = np.asarray([[0.8, 0.2]])
    combined = weighted_probability_average([first, second], [1, 3])
    assert np.allclose(combined, [[0.65, 0.35]])


def test_weighted_probability_average_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shapes differ"):
        weighted_probability_average([np.zeros((2, 2)), np.zeros((2, 3))])


def test_percentile_ranks_average_ties() -> None:
    values = np.asarray([[0.2], [0.2], [0.8]])

    ranked = percentile_ranks(values)

    assert np.allclose(ranked[:, 0], [0.5, 0.5, 1.0])


def test_weighted_rank_average_ignores_probability_scale() -> None:
    first = np.asarray([[0.1], [0.2], [0.9]])
    second = np.asarray([[0.6], [0.7], [0.8]])

    combined = weighted_rank_average([first, second])

    assert np.allclose(combined[:, 0], [1 / 3, 2 / 3, 1.0])
