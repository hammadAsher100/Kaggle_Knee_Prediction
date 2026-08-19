"""Tabular competition schema discovery and integrity audit helpers."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

IDENTIFIER_TOKENS = ("id", "uid", "study", "patient", "exam", "case", "instance")


def _looks_like_identifier(column: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", column.lower()).strip("_")
    parts = set(normalized.split("_"))
    return (
        bool(parts.intersection(IDENTIFIER_TOKENS))
        or normalized.endswith("uid")
        or normalized.endswith("instanceuid")
        or normalized in {"studyinstanceuid", "seriesinstanceuid", "sopinstanceuid"}
    )


def read_table(path: str | Path) -> pd.DataFrame:
    """Read a supported competition table from disk."""
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Table does not exist: {resolved}")
    suffix = resolved.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(resolved)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(resolved)
    if suffix in {".json", ".jsonl"}:
        return pd.read_json(resolved, lines=suffix == ".jsonl")
    raise ValueError(f"Unsupported table suffix: {suffix}")


def infer_submission_schema(
    train: pd.DataFrame,
    sample_submission: pd.DataFrame,
) -> dict[str, Any]:
    """Infer ID and target columns only when the sample schema is unambiguous."""
    if sample_submission.shape[1] < 2:
        raise ValueError("Sample submission must contain an identifier and at least one target")
    named_identifiers = [
        column
        for column in sample_submission.columns
        if _looks_like_identifier(column)
    ]
    if len(named_identifiers) == 1:
        identifier_candidates = named_identifiers
    elif len(named_identifiers) > 1:
        raise ValueError(
            "Unable to infer exactly one submission identifier; configure it explicitly"
        )
    else:
        identifier_candidates = []
    target_candidates = [
        column
        for column in sample_submission.columns
        if column not in named_identifiers
        if pd.api.types.is_numeric_dtype(sample_submission[column])
    ]
    if not identifier_candidates:
        non_targets = [
            column for column in sample_submission.columns if column not in target_candidates
        ]
        if len(non_targets) == 1:
            identifier_candidates = non_targets
    if len(identifier_candidates) != 1:
        raise ValueError(
            "Unable to infer exactly one submission identifier; configure it explicitly"
        )
    identifier = identifier_candidates[0]
    targets = [column for column in sample_submission.columns if column in target_candidates]
    if not targets:
        raise ValueError("Unable to infer numeric target columns from sample submission")
    extra_submission_columns = [
        column for column in sample_submission.columns if column not in {identifier, *targets}
    ]
    if extra_submission_columns:
        raise ValueError(
            "Unclassified sample submission columns: " + ", ".join(extra_submission_columns)
        )
    return {
        "row_identifier": identifier,
        "target_columns": targets,
        "submission_columns": list(sample_submission.columns),
    }


def identifier_profile(frame: pd.DataFrame, columns: Sequence[str]) -> dict[str, Any]:
    """Profile uniqueness, missingness, and cardinality for candidate IDs."""
    profiles: dict[str, Any] = {}
    for column in columns:
        if column not in frame:
            continue
        series = frame[column]
        profiles[column] = {
            "dtype": str(series.dtype),
            "row_count": int(len(series)),
            "non_null_count": int(series.notna().sum()),
            "unique_non_null_count": int(series.nunique(dropna=True)),
            "missing_count": int(series.isna().sum()),
            "duplicate_non_null_rows": int(series.dropna().duplicated(keep=False).sum()),
        }
    return profiles


def target_audit(frame: pd.DataFrame, targets: Sequence[str]) -> dict[str, Any]:
    """Compute target domains, prevalence, and co-occurrence without coercion."""
    missing = [column for column in targets if column not in frame]
    if missing:
        raise ValueError(f"Target columns missing from training data: {', '.join(missing)}")
    target_frame = frame.loc[:, targets]
    prevalence: dict[str, Any] = {}
    for target in targets:
        values = target_frame[target]
        known = values.dropna()
        prevalence[target] = {
            "dtype": str(values.dtype),
            "missing_count": int(values.isna().sum()),
            "known_count": int(values.notna().sum()),
            "unique_values": [
                value.item() if isinstance(value, np.generic) else value
                for value in values.dropna().unique().tolist()
            ],
            "positive_count": int((values == 1).sum()),
            "positive_prevalence_among_known": (
                float(known.eq(1).mean()) if not known.empty else None
            ),
        }
    binary = target_frame.eq(1).astype(np.int64)
    cooccurrence = binary.T.dot(binary)
    return {
        "prevalence": prevalence,
        "cooccurrence": {
            row: {column: int(cooccurrence.loc[row, column]) for column in targets}
            for row in targets
        },
    }


def audit_competition_tables(
    train_path: str | Path,
    sample_submission_path: str | Path,
    *,
    configured_schema: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Audit exact table schemas and derive targets from the sample submission."""
    train = read_table(train_path)
    sample = read_table(sample_submission_path)
    inferred = infer_submission_schema(train, sample)
    schema = dict(inferred)
    if configured_schema:
        for key, value in configured_schema.items():
            if value not in (None, [], "auto"):
                schema[key] = value
    targets = list(schema["target_columns"])
    missing_training_targets = [target for target in targets if target not in train]
    targets_result: dict[str, Any]
    if missing_training_targets:
        targets_result = {
            "available_in_training_table": False,
            "missing_training_targets": missing_training_targets,
            "prevalence": None,
            "cooccurrence": None,
        }
    else:
        targets_result = {
            "available_in_training_table": True,
            "missing_training_targets": [],
            **target_audit(train, targets),
        }
    candidate_ids = [
        column
        for column in train.columns
        if _looks_like_identifier(column)
    ]
    report_candidates = [
        column
        for column in train.columns
        if any(token in column.lower() for token in ("report", "text", "findings", "impression"))
    ]
    entity_counts = {
        key: (
            int(train[value].nunique(dropna=True))
            if value is not None and value in train
            else None
        )
        for key, value in {
            "row_identifier_count": schema.get("row_identifier"),
            "study_count": schema.get("study_identifier"),
            "patient_count": schema.get("patient_identifier"),
            "institution_count": schema.get("institution_identifier"),
        }.items()
    }
    return {
        "train_path": str(Path(train_path).expanduser().resolve()),
        "sample_submission_path": str(Path(sample_submission_path).expanduser().resolve()),
        "train_row_count": int(len(train)),
        "sample_submission_row_count": int(len(sample)),
        "train_columns": list(train.columns),
        "train_dtypes": {column: str(dtype) for column, dtype in train.dtypes.items()},
        "sample_submission_columns": list(sample.columns),
        "schema": schema,
        "entity_counts": entity_counts,
        "identifier_candidates": candidate_ids,
        "identifier_profiles": identifier_profile(train, candidate_ids),
        "report_column_candidates": report_candidates,
        "duplicate_train_rows": int(train.duplicated().sum()),
        "duplicate_submission_rows": int(sample.duplicated().sum()),
        "column_missingness": {
            column: {
                "missing_count": int(train[column].isna().sum()),
                "missing_fraction": float(train[column].isna().mean()),
            }
            for column in train.columns
        },
        "targets": targets_result,
    }


def write_audit_json(audit: Mapping[str, Any], path: str | Path) -> Path:
    """Write an audit atomically to avoid leaving a partial result."""
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    temporary.replace(output)
    return output
