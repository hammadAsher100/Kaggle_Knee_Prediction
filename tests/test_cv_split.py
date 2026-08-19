"""Tests for deterministic multilabel group-safe folds."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.training.cv_split import (
    assert_group_disjoint,
    assert_train_valid_disjoint,
    fold_audit,
    make_multilabel_group_folds,
)


def _synthetic_studies() -> pd.DataFrame:
    rows = []
    for patient_index in range(30):
        for study_offset in range(1 + (patient_index % 3 == 0)):
            study_index = len(rows)
            rows.append(
                {
                    "patient_id": f"patient-{patient_index:02d}",
                    "study_id": f"study-{study_index:02d}",
                    "target_a": int(patient_index % 3 == 0),
                    "target_b": int(patient_index % 5 == 0 or study_offset == 1),
                    "site": f"site-{patient_index % 3}",
                }
            )
    return pd.DataFrame(rows)


def test_cv_has_disjoint_patients_and_studies_and_is_deterministic() -> None:
    frame = _synthetic_studies()
    first, first_quality = make_multilabel_group_folds(
        frame,
        group_column="patient_id",
        target_columns=["target_a", "target_b"],
        n_splits=5,
        seed=17,
        restarts=16,
    )
    second, second_quality = make_multilabel_group_folds(
        frame,
        group_column="patient_id",
        target_columns=["target_a", "target_b"],
        n_splits=5,
        seed=17,
        restarts=16,
    )

    assert first["fold"].tolist() == second["fold"].tolist()
    assert first_quality == second_quality
    assert set(first["fold"]) == set(range(5))
    assert_group_disjoint(first, group_column="patient_id")
    for fold in range(5):
        assert_train_valid_disjoint(
            first,
            validation_fold=fold,
            study_column="study_id",
            patient_column="patient_id",
        )
    positive_counts = first.groupby("fold")[["target_a", "target_b"]].sum()
    assert np.ptp(positive_counts["target_a"].to_numpy()) <= 2
    assert np.ptp(positive_counts["target_b"].to_numpy()) <= 2


def test_fold_audit_includes_targets_patients_and_subgroups() -> None:
    frame, _ = make_multilabel_group_folds(
        _synthetic_studies(),
        group_column="patient_id",
        target_columns=["target_a", "target_b"],
        n_splits=3,
        restarts=8,
    )

    audit = fold_audit(
        frame,
        target_columns=["target_a", "target_b"],
        study_column="study_id",
        patient_column="patient_id",
        subgroup_columns=["site"],
    )

    assert set(audit["folds"]) == {"0", "1", "2"}
    assert sum(item["study_count"] for item in audit["folds"].values()) == len(frame)
    assert all(item["patient_count"] for item in audit["folds"].values())
    assert all("site" in item["subgroups"] for item in audit["folds"].values())


def test_sparse_labels_balance_coverage_without_treating_unknown_as_negative() -> None:
    frame = _synthetic_studies()
    frame.loc[frame.index % 3 != 0, "target_a"] = np.nan
    folded, _ = make_multilabel_group_folds(
        frame,
        group_column="patient_id",
        target_columns=["target_a", "target_b"],
        n_splits=3,
        restarts=16,
    )

    audit = fold_audit(
        folded,
        target_columns=["target_a", "target_b"],
        study_column="study_id",
        patient_column="patient_id",
    )

    known_counts = [item["targets"]["target_a"]["known_count"] for item in audit["folds"].values()]
    assert max(known_counts) - min(known_counts) <= 1
    assert sum(known_counts) == int(frame["target_a"].notna().sum())
    for item in audit["folds"].values():
        target = item["targets"]["target_a"]
        assert target["known_count"] == target["positive_count"] + target["negative_count"]


def test_cv_rejects_missing_groups_nonbinary_labels_and_too_many_folds() -> None:
    frame = _synthetic_studies()
    with pytest.raises(ValueError, match="missing values"):
        make_multilabel_group_folds(
            frame.assign(patient_id=None),
            group_column="patient_id",
            target_columns=["target_a"],
        )
    with pytest.raises(ValueError, match="not binary"):
        make_multilabel_group_folds(
            frame.assign(target_a=2),
            group_column="patient_id",
            target_columns=["target_a"],
        )
    with pytest.raises(ValueError, match="unique groups"):
        make_multilabel_group_folds(
            frame,
            group_column="patient_id",
            target_columns=["target_a"],
            n_splits=31,
        )
