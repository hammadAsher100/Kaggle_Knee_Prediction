"""Tests for constrained probability averaging."""

import numpy as np
import pytest

from src.inference.ensemble import weighted_probability_average


def test_weighted_probability_average_normalizes_weights() -> None:
    first = np.asarray([[0.2, 0.8]])
    second = np.asarray([[0.8, 0.2]])
    combined = weighted_probability_average([first, second], [1, 3])
    assert np.allclose(combined, [[0.65, 0.35]])


def test_weighted_probability_average_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shapes differ"):
        weighted_probability_average([np.zeros((2, 2)), np.zeros((2, 3))])
