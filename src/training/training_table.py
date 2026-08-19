"""Build leakage-safe training targets and fold proxies from weak report labels."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd

from src.training.cv_split import fold_audit, make_multilabel_group_folds


def choose_group_column(
    inventory: pd.DataFrame,
    *,
    study_column: str = "StudyInstanceUID",
    patient_column: str = "PatientID",
) -> tuple[pd.Series, str]:
    """Use recoverable patient groups; otherwise fall back explicitly to studies."""
    studies = inventory[study_column].astype(str)
    if patient_column not in inventory:
        return studies, study_column
    patients = inventory[patient_column].astype("string").str.strip()
    usable = patients.notna() & patients.ne("")
    shared = patients[usable].duplicated(keep=False)
    if usable.all() and shared.any():
        return patients.astype(str), patient_column
    return studies, study_column


def _prevalence_rank_proxy(probabilities: pd.Series, gold: pd.Series) -> pd.Series:
    known = gold.dropna().astype(float)
    prevalence = float(known.mean()) if not known.empty else 0.5
    prevalence = min(max(prevalence, 0.02), 0.98)
    rank = probabilities.astype(float).rank(method="first", pct=True)
    proxy = rank.ge(1.0 - prevalence).astype("int8")
    proxy.loc[gold.notna()] = gold.loc[gold.notna()].astype("int8")
    return proxy


def build_training_table(
    train: pd.DataFrame,
    semantic_labels: pd.DataFrame,
    inventory: pd.DataFrame,
    *,
    target_columns: Sequence[str],
    study_column: str = "StudyInstanceUID",
    n_splits: int = 5,
    seed: int = 20260812,
    restarts: int = 64,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Merge targets, override weak labels with gold, and assign grouped folds."""
    inputs = (
        ("train", train),
        ("semantic_labels", semantic_labels),
        ("inventory", inventory),
    )
    for name, frame in inputs:
        if study_column not in frame:
            raise ValueError(f"{name} is missing {study_column}")
        if frame[study_column].duplicated().any():
            raise ValueError(f"{name} contains duplicate studies")
    merged = train[[study_column, "PatientSex", *target_columns]].merge(
        semantic_labels,
        on=study_column,
        how="left",
        validate="one_to_one",
    ).merge(inventory, on=study_column, how="left", validate="one_to_one")
    if len(merged) != len(train):
        raise AssertionError("Training-table merge changed row count")
    proxy_columns: list[str] = []
    for target in target_columns:
        weak_column = f"{target}__probability"
        if weak_column not in merged:
            raise ValueError(f"Semantic labels are missing {weak_column}")
        gold = pd.to_numeric(merged[target], errors="coerce")
        weak = pd.to_numeric(merged[weak_column], errors="coerce")
        if weak.isna().any() or not weak.between(0, 1).all():
            raise ValueError(f"Weak probabilities are invalid for {target}")
        merged[f"{target}__gold"] = gold
        merged[f"{target}__gold_mask"] = gold.notna().astype("int8")
        merged[f"{target}__weak"] = weak.astype("float32")
        merged[f"{target}__train"] = gold.fillna(weak).astype("float32")
        proxy_column = f"{target}__stratify"
        merged[proxy_column] = _prevalence_rank_proxy(weak, gold)
        proxy_columns.append(proxy_column)
    groups, group_source = choose_group_column(merged, study_column=study_column)
    merged["leakage_group"] = groups
    folded, quality = make_multilabel_group_folds(
        merged,
        group_column="leakage_group",
        target_columns=proxy_columns,
        n_splits=n_splits,
        seed=seed,
        restarts=restarts,
    )
    audit = fold_audit(
        folded,
        target_columns=[f"{target}__gold" for target in target_columns],
        study_column=study_column,
        patient_column="leakage_group" if group_source == "PatientID" else None,
        subgroup_columns=["PatientSex"],
    )
    audit.update(
        {
            "grouping_source": group_source,
            "quality": quality.to_dict(),
            "gold_study_count": int(
                folded[[f"{target}__gold_mask" for target in target_columns]]
                .max(axis=1)
                .sum()
            ),
            "weak_label_prevalence": {
                target: float(folded[f"{target}__weak"].mean())
                for target in target_columns
            },
            "train_target_prevalence": {
                target: float(folded[f"{target}__train"].mean())
                for target in target_columns
            },
        }
    )
    return folded, audit
