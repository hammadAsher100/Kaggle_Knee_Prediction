"""Per-target and macro ROC AUC with missing-label and degenerate-fold handling."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class AUCResult:
    macro_auc: float | None
    per_target: dict[str, float | None]
    valid_target_count: int
    target_count: int
    known_count: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def multilabel_roc_auc(
    targets: np.ndarray,
    probabilities: np.ndarray,
    target_names: Sequence[str],
) -> AUCResult:
    """Compute target AUCs, excluding missing rows and single-class targets."""
    from sklearn.metrics import roc_auc_score

    truth = np.asarray(targets, dtype=float)
    scores = np.asarray(probabilities, dtype=float)
    if truth.shape != scores.shape or truth.ndim != 2:
        raise ValueError("targets and probabilities must be same-shaped 2D arrays")
    if truth.shape[1] != len(target_names):
        raise ValueError("target_names does not match array width")
    if not np.isfinite(scores).all():
        raise ValueError("probabilities contain non-finite values")
    if ((scores < 0) | (scores > 1)).any():
        raise ValueError("probabilities must lie in [0, 1]")
    per_target: dict[str, float | None] = {}
    known_count: dict[str, int] = {}
    valid: list[float] = []
    for index, name in enumerate(target_names):
        mask = np.isfinite(truth[:, index])
        known_count[name] = int(mask.sum())
        y_true = truth[mask, index]
        if len(np.unique(y_true)) < 2:
            per_target[name] = None
            continue
        auc = float(roc_auc_score(y_true, scores[mask, index]))
        per_target[name] = auc
        valid.append(auc)
    return AUCResult(
        macro_auc=float(np.mean(valid)) if valid else None,
        per_target=per_target,
        valid_target_count=len(valid),
        target_count=len(target_names),
        known_count=known_count,
    )
