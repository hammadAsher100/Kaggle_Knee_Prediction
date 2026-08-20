"""Validation and normalization for public report-derived label tables."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def normalize_external_labels(
    train: pd.DataFrame,
    external: pd.DataFrame,
    target_columns: Sequence[str],
    *,
    study_column: str = "StudyInstanceUID",
) -> pd.DataFrame:
    """Convert a complete public label table to the internal semantic-label schema."""
    required = {study_column, *target_columns}
    for name, frame in (("train", train), ("external", external)):
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{name} is missing columns: {sorted(missing)}")
        if frame[study_column].isna().any() or frame[study_column].duplicated().any():
            raise ValueError(f"{name} study IDs must be non-missing and unique")

    train_ids = train[study_column].astype(str)
    external_ids = external[study_column].astype(str)
    if set(train_ids) != set(external_ids):
        raise ValueError(
            "external study IDs differ from train: "
            f"missing={len(set(train_ids) - set(external_ids))}, "
            f"extra={len(set(external_ids) - set(train_ids))}"
        )

    aligned = pd.DataFrame({study_column: train_ids}).merge(
        external.assign(**{study_column: external_ids}),
        on=study_column,
        how="left",
        validate="one_to_one",
        sort=False,
    )
    output = aligned[[study_column]].copy()
    for target in target_columns:
        values = pd.to_numeric(aligned[target], errors="coerce").to_numpy(float)
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError(f"external probabilities are invalid for {target}")
        output[f"{target}__semantic_probability"] = values
        output[f"{target}__rule_probability"] = values
        output[f"{target}__probability"] = values
    return output
