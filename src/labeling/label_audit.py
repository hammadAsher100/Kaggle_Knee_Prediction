"""Aggregate weak-label coverage and gold-label validation metrics."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd
from sklearn.metrics import roc_auc_score


def audit_weak_labels(
    labels: pd.DataFrame,
    *,
    target_columns: Sequence[str],
) -> dict[str, Any]:
    targets: dict[str, Any] = {}
    auc_values: list[float] = []
    for target in target_columns:
        gold = labels[f"{target}__gold"]
        probability = labels[f"{target}__weak_probability"]
        status = labels[f"{target}__weak_status"]
        known = gold.notna()
        auc: float | None = None
        if known.any() and gold[known].nunique() == 2:
            auc = float(roc_auc_score(gold[known].astype(int), probability[known]))
            auc_values.append(auc)
        determinate = labels[f"{target}__weak_label"].notna()
        accuracy: float | None = None
        evaluated = known & determinate
        if evaluated.any():
            accuracy = float(
                (
                    labels.loc[evaluated, f"{target}__weak_label"].astype(int)
                    == gold[evaluated].astype(int)
                ).mean()
            )
        targets[target] = {
            "gold_known": int(known.sum()),
            "weak_determinate": int(determinate.sum()),
            "weak_coverage": float(determinate.mean()),
            "gold_auc": auc,
            "gold_determinate_accuracy": accuracy,
            "status_counts": {str(key): int(value) for key, value in status.value_counts().items()},
        }
    return {
        "row_count": int(len(labels)),
        "macro_gold_auc": float(sum(auc_values) / len(auc_values)) if auc_values else None,
        "language_counts": {
            str(key): int(value) for key, value in labels["language"].value_counts().items()
        },
        "targets": targets,
    }
