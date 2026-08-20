"""Leakage-safe nested blending for multimodal out-of-fold predictions."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
from sklearn.metrics import roc_auc_score


@dataclass(frozen=True)
class NestedFusionResult:
    """Predictions and diagnostics from nested simplex weight selection."""

    predictions: np.ndarray
    weights: dict[int, np.ndarray]
    macro_auc: float
    per_target_auc: np.ndarray


def simplex_weights(n_modalities: int, step: float) -> np.ndarray:
    """Return non-negative weight vectors on a discrete probability simplex."""
    if n_modalities < 2:
        raise ValueError("n_modalities must be at least 2")
    if not 0 < step <= 1:
        raise ValueError("step must be in (0, 1]")

    divisions = round(1 / step)
    if not np.isclose(divisions * step, 1.0):
        raise ValueError("step must divide 1 exactly")

    integer_weights = [
        values
        for values in product(range(divisions + 1), repeat=n_modalities)
        if sum(values) == divisions
    ]
    return np.asarray(integer_weights, dtype=np.float64) / divisions


def _target_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    valid = np.isfinite(y_true) & np.isfinite(y_score)
    labels = y_true[valid]
    if labels.size == 0 or np.unique(labels).size < 2:
        return float("nan")
    return float(roc_auc_score(labels, y_score[valid]))


def score_targets(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, np.ndarray]:
    """Calculate macro and per-target AUC while ignoring missing gold labels."""
    if y_true.shape != y_score.shape or y_true.ndim != 2:
        raise ValueError("y_true and y_score must be matching two-dimensional arrays")
    per_target = np.asarray(
        [_target_auc(y_true[:, index], y_score[:, index]) for index in range(y_true.shape[1])]
    )
    return float(np.nanmean(per_target)), per_target


def nested_simplex_fusion(
    y_true: np.ndarray,
    predictions: np.ndarray,
    folds: np.ndarray,
    *,
    step: float = 0.05,
    per_target: bool = False,
) -> NestedFusionResult:
    """Select blend weights away from each held-out fold and score nested OOF predictions.

    ``predictions`` has shape ``(samples, targets, modalities)``. Gold labels may be
    missing, but each held-out fold must contain at least one gold row to contribute to
    the final score. In global mode one weight vector is selected across all targets;
    per-target mode selects a separate vector for every target.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)
    folds = np.asarray(folds)
    if y_true.ndim != 2:
        raise ValueError("y_true must be two-dimensional")
    if predictions.shape[:2] != y_true.shape or predictions.ndim != 3:
        raise ValueError("predictions must have shape (samples, targets, modalities)")
    if folds.shape != (y_true.shape[0],):
        raise ValueError("folds must contain one value per sample")
    if not np.isfinite(predictions).all():
        raise ValueError("predictions must be finite")

    candidates = simplex_weights(predictions.shape[2], step)
    fused = np.full(y_true.shape, np.nan, dtype=np.float64)
    selected: dict[int, np.ndarray] = {}

    for fold_value in np.unique(folds):
        train_mask = folds != fold_value
        valid_mask = folds == fold_value
        if not np.isfinite(y_true[train_mask]).any():
            raise ValueError(f"fold {fold_value!r} has no training gold labels")

        if per_target:
            fold_weights = np.empty((y_true.shape[1], predictions.shape[2]))
            for target_index in range(y_true.shape[1]):
                scores = [
                    _target_auc(
                        y_true[train_mask, target_index],
                        np.einsum(
                            "sm,m->s",
                            predictions[train_mask, target_index, :],
                            candidate,
                        ),
                    )
                    for candidate in candidates
                ]
                if np.isnan(scores).all():
                    raise ValueError(
                        f"target {target_index} has fewer than two classes outside "
                        f"fold {fold_value!r}"
                    )
                fold_weights[target_index] = candidates[int(np.nanargmax(scores))]
            fused[valid_mask] = np.einsum(
                "stm,tm->st", predictions[valid_mask], fold_weights
            )
        else:
            scores = [
                score_targets(
                    y_true[train_mask],
                    np.einsum("stm,m->st", predictions[train_mask], candidate),
                )[0]
                for candidate in candidates
            ]
            fold_weights = candidates[int(np.nanargmax(scores))]
            fused[valid_mask] = np.einsum(
                "stm,m->st", predictions[valid_mask], fold_weights
            )
        selected[int(fold_value)] = fold_weights

    macro_auc, per_target_auc = score_targets(y_true, fused)
    return NestedFusionResult(fused, selected, macro_auc, per_target_auc)
