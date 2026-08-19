"""Tests for weak/gold target construction and grouping fallback."""

import numpy as np
import pandas as pd

from src.training.training_table import build_training_table, choose_group_column


def test_training_table_overrides_weak_targets_and_groups_patients() -> None:
    studies = [f"s{index}" for index in range(10)]
    train = pd.DataFrame(
        {
            "StudyInstanceUID": studies,
            "PatientSex": ["Male", "Female"] * 5,
            "ACL": [1.0, 0.0] + [np.nan] * 8,
        }
    )
    labels = pd.DataFrame(
        {"StudyInstanceUID": studies, "ACL__probability": np.linspace(0.05, 0.95, 10)}
    )
    inventory = pd.DataFrame(
        {
            "StudyInstanceUID": studies,
            "PatientID": [f"p{index // 2}" for index in range(10)],
        }
    )
    folded, audit = build_training_table(
        train,
        labels,
        inventory,
        target_columns=["ACL"],
        n_splits=2,
        restarts=4,
    )
    assert folded.loc[0, "ACL__train"] == 1.0
    assert folded.loc[1, "ACL__train"] == 0.0
    assert folded.loc[2, "ACL__train"] == folded.loc[2, "ACL__weak"]
    assert folded.groupby("PatientID")["fold"].nunique().max() == 1
    assert audit["grouping_source"] == "PatientID"


def test_grouping_falls_back_if_patient_ids_are_unusable() -> None:
    inventory = pd.DataFrame(
        {"StudyInstanceUID": ["a", "b"], "PatientID": [None, None]}
    )
    groups, source = choose_group_column(inventory)
    assert source == "StudyInstanceUID"
    assert groups.tolist() == ["a", "b"]
