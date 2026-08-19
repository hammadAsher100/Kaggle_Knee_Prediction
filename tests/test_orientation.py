"""Tests for geometry-derived orientation."""

import numpy as np

from src.data.orientation import geometry_order, infer_anatomical_plane, slice_normal


def test_orientation_normalization_contract() -> None:
    assert np.allclose(slice_normal([1, 0, 0, 0, 1, 0]), [0, 0, 1])
    assert infer_anatomical_plane([1, 0, 0, 0, 1, 0]) == "Axial"
    assert infer_anatomical_plane(None) == "Unknown"
    order, method = geometry_order([3.0, 1.0, 2.0], [30, 10, 20])
    assert method == "geometry"
    assert order.tolist() == [1, 2, 0]
    order, method = geometry_order([None, None], [2, 1])
    assert method == "instance_number"
    assert order.tolist() == [1, 0]
