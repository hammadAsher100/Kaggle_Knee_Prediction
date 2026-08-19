"""Tests for deterministic 2.5D slice sampling."""

import numpy as np

from src.data.slice_sampler import neighborhood_indices, random_uniform_indices, uniform_indices


def test_slice_sampler_boundary_contract() -> None:
    assert uniform_indices(2, 5).tolist() == [0, 0, 0, 1, 1]
    assert neighborhood_indices(0, 3).tolist() == [0, 0, 1]
    first = random_uniform_indices(20, 5, np.random.default_rng(7))
    second = random_uniform_indices(20, 5, np.random.default_rng(7))
    assert first.tolist() == second.tolist()
