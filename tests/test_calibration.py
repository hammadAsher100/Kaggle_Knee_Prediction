"""Tests for conservative monotonic probability calibration."""

import numpy as np

from src.inference.calibration import (
    apply_multilabel_calibration,
    apply_platt,
    fit_monotonic_platt,
)


def test_platt_transform_is_monotonic_and_finite() -> None:
    labels = np.asarray([0, 0, 0, 1, 1, 1], dtype=float)
    probabilities = np.asarray([0.05, 0.15, 0.3, 0.7, 0.8, 0.95])
    parameters = fit_monotonic_platt(labels, probabilities)
    transformed = apply_platt(probabilities, parameters)
    assert np.isfinite(transformed).all()
    assert np.all(np.diff(transformed) >= 0)
    assert np.logical_and(transformed > 0, transformed < 1).all()


def test_multilabel_calibration_requires_every_target() -> None:
    values = np.asarray([[0.25, 0.75]])
    parameters = {
        "a": {"slope": 1.0, "intercept": 0.0},
        "b": {"slope": 1.0, "intercept": 0.0},
    }
    calibrated = apply_multilabel_calibration(values, ["a", "b"], parameters)
    assert np.allclose(calibrated, values)
