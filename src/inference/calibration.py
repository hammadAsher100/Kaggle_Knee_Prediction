"""Conservative monotonic Platt calibration for multilabel probabilities."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def probability_logit(probabilities: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(probabilities, dtype=float), 1e-5, 1.0 - 1e-5)
    return np.log(values / (1.0 - values))


def fit_monotonic_platt(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    """Fit positive-slope logit scaling, or identity when evidence is insufficient."""
    from scipy.optimize import minimize

    truth = np.asarray(labels, dtype=float)
    scores = np.asarray(probabilities, dtype=float)
    mask = np.isfinite(truth) & np.isfinite(scores)
    truth = truth[mask]
    scores = scores[mask]
    positives = int((truth == 1).sum())
    negatives = int((truth == 0).sum())
    if positives < 3 or negatives < 3:
        return {
            "slope": 1.0,
            "intercept": 0.0,
            "status": "identity_insufficient_classes",
            "known_count": len(truth),
        }
    logits = probability_logit(scores)

    def objective(parameters: np.ndarray) -> float:
        transformed = parameters[0] * logits + parameters[1]
        losses = np.logaddexp(0.0, transformed) - truth * transformed
        regularization = 0.01 * ((parameters[0] - 1.0) ** 2 + parameters[1] ** 2)
        return float(losses.mean() + regularization)

    result = minimize(
        objective,
        np.asarray([1.0, 0.0]),
        method="L-BFGS-B",
        bounds=((0.05, 10.0), (-6.0, 6.0)),
    )
    if not result.success:
        return {
            "slope": 1.0,
            "intercept": 0.0,
            "status": "identity_optimizer_failure",
            "known_count": len(truth),
        }
    return {
        "slope": float(result.x[0]),
        "intercept": float(result.x[1]),
        "status": "fitted",
        "known_count": len(truth),
    }


def apply_platt(probabilities: np.ndarray, parameters: dict[str, Any]) -> np.ndarray:
    transformed = (
        float(parameters["slope"]) * probability_logit(probabilities)
        + float(parameters["intercept"])
    )
    return 1.0 / (1.0 + np.exp(-np.clip(transformed, -30.0, 30.0)))


def apply_multilabel_calibration(
    probabilities: np.ndarray,
    target_names: Sequence[str],
    parameters: dict[str, dict[str, Any]],
) -> np.ndarray:
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(target_names):
        raise ValueError("Probability matrix does not match target names")
    calibrated = values.copy()
    for index, target in enumerate(target_names):
        if target not in parameters:
            raise ValueError(f"Calibration is missing target {target}")
        calibrated[:, index] = apply_platt(values[:, index], parameters[target])
    return calibrated
