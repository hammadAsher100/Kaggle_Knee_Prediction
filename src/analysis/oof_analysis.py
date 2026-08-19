"""OOF uncertainty and subgroup analysis."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from src.training.metrics import multilabel_roc_auc


def bootstrap_macro_auc(
    truth: np.ndarray,
    probabilities: np.ndarray,
    target_names: Sequence[str],
    *,
    iterations: int = 1000,
    seed: int = 20260812,
) -> dict[str, Any]:
    """Study-level bootstrap interval with invalid resamples excluded."""
    if iterations < 1:
        raise ValueError("iterations must be positive")
    point = multilabel_roc_auc(truth, probabilities, target_names)
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(iterations):
        indices = rng.integers(0, len(truth), len(truth))
        result = multilabel_roc_auc(truth[indices], probabilities[indices], target_names)
        if result.macro_auc is not None:
            values.append(result.macro_auc)
    return {
        "point_estimate": point.macro_auc,
        "lower_95": float(np.percentile(values, 2.5)) if values else None,
        "upper_95": float(np.percentile(values, 97.5)) if values else None,
        "valid_bootstraps": len(values),
        "requested_bootstraps": iterations,
    }


def subgroup_auc(
    frame: pd.DataFrame,
    target_names: Sequence[str],
    *,
    subgroup_column: str,
    prediction_suffix: str = "__prediction",
) -> dict[str, Any]:
    """Report subgroup AUC only where both classes are present."""
    result: dict[str, Any] = {}
    for subgroup, subset in frame.groupby(subgroup_column, dropna=False, sort=True):
        truth = subset[[f"{target}__gold" for target in target_names]].to_numpy(float)
        probabilities = subset[
            [f"{target}{prediction_suffix}" for target in target_names]
        ].to_numpy(float)
        metrics = multilabel_roc_auc(truth, probabilities, target_names)
        result[str(subgroup)] = {"row_count": len(subset), **metrics.to_dict()}
    return result
