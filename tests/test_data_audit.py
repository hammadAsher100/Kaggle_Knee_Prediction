"""Tests for exact competition table schema discovery and auditing."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data.audit import audit_competition_tables, infer_submission_schema


def test_infer_schema_supports_numeric_identifier_and_preserves_target_order() -> None:
    train = pd.DataFrame(
        {
            "study_id": [101, 102, 103],
            "report_text": ["a", "b", "c"],
            "tear": [0, 1, 0],
            "effusion": [1, 0, 1],
        }
    )
    sample = pd.DataFrame(
        {
            "study_id": [201, 202],
            "effusion": [0.5, 0.5],
            "tear": [0.5, 0.5],
        }
    )

    schema = infer_submission_schema(train, sample)

    assert schema["row_identifier"] == "study_id"
    assert schema["target_columns"] == ["effusion", "tear"]
    assert schema["submission_columns"] == ["study_id", "effusion", "tear"]


def test_audit_tables_reports_prevalence_cooccurrence_and_duplicates(tmp_path: Path) -> None:
    train_path = tmp_path / "train.csv"
    sample_path = tmp_path / "sample_submission.csv"
    pd.DataFrame(
        {
            "StudyInstanceUID": ["a", "b", "b"],
            "report": ["normal", "tear", "tear"],
            "target_x": [0, 1, 1],
            "target_y": [1, 1, 1],
        }
    ).to_csv(train_path, index=False)
    pd.DataFrame(
        {
            "StudyInstanceUID": ["c", "d"],
            "target_x": [0.5, 0.5],
            "target_y": [0.5, 0.5],
        }
    ).to_csv(sample_path, index=False)

    audit = audit_competition_tables(train_path, sample_path)

    assert audit["train_row_count"] == 3
    assert audit["schema"]["target_columns"] == ["target_x", "target_y"]
    assert audit["identifier_profiles"]["StudyInstanceUID"]["unique_non_null_count"] == 2
    assert audit["targets"]["available_in_training_table"] is True
    assert audit["targets"]["prevalence"]["target_x"]["positive_count"] == 2
    assert audit["targets"]["prevalence"]["target_x"]["known_count"] == 3
    assert audit["targets"]["cooccurrence"]["target_x"]["target_y"] == 2
    assert audit["report_column_candidates"] == ["report"]


def test_infer_schema_rejects_ambiguous_identifiers() -> None:
    train = pd.DataFrame({"study_id": [1], "patient_id": [2], "target": [1]})
    sample = pd.DataFrame({"study_id": [3], "patient_id": [4], "target": [0.5]})

    with pytest.raises(ValueError, match="exactly one"):
        infer_submission_schema(train, sample)


def test_audit_discovers_submission_targets_absent_from_training(tmp_path: Path) -> None:
    train_path = tmp_path / "train.csv"
    sample_path = tmp_path / "sample_submission.csv"
    pd.DataFrame(
        {"study_id": [1, 2], "report_text": ["normal", "possible tear"]}
    ).to_csv(train_path, index=False)
    pd.DataFrame({"study_id": [3], "tear": [0.5], "effusion": [0.5]}).to_csv(
        sample_path,
        index=False,
    )

    audit = audit_competition_tables(train_path, sample_path)

    assert audit["schema"]["target_columns"] == ["tear", "effusion"]
    assert audit["targets"]["available_in_training_table"] is False
    assert audit["targets"]["missing_training_targets"] == ["tear", "effusion"]
