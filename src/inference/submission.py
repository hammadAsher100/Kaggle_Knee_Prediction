"""Strict competition submission construction and validation."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd


def validate_submission(
    submission: pd.DataFrame,
    sample_submission: pd.DataFrame,
    *,
    identifier: str = "StudyInstanceUID",
) -> None:
    """Raise on any schema, ID, ordering, duplication, or probability defect."""
    if list(submission.columns) != list(sample_submission.columns):
        raise ValueError("Submission columns or order do not match sample submission")
    if len(submission) != len(sample_submission):
        raise ValueError("Submission row count does not match sample submission")
    if identifier not in submission:
        raise ValueError(f"Missing identifier column: {identifier}")
    if submission[identifier].isna().any() or submission[identifier].duplicated().any():
        raise ValueError("Submission identifiers are missing or duplicated")
    submission_ids = submission[identifier].astype(str).tolist()
    sample_ids = sample_submission[identifier].astype(str).tolist()
    if submission_ids != sample_ids:
        raise ValueError("Submission identifiers or order do not match sample submission")
    target_columns = [column for column in submission if column != identifier]
    values = submission[target_columns].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Submission contains non-finite predictions")
    if ((values < 0.0) | (values > 1.0)).any():
        raise ValueError("Submission predictions must lie in [0, 1]")


def build_submission(
    study_ids: Sequence[str],
    probabilities: np.ndarray,
    sample_submission: pd.DataFrame,
    *,
    identifier: str = "StudyInstanceUID",
) -> pd.DataFrame:
    """Align predictions by study ID to the immutable sample-submission order."""
    target_columns = [column for column in sample_submission if column != identifier]
    values = np.asarray(probabilities, dtype=float)
    if values.shape != (len(study_ids), len(target_columns)):
        raise ValueError("Prediction shape does not match study IDs and target columns")
    predictions = pd.DataFrame(values, columns=target_columns)
    predictions.insert(0, identifier, [str(value) for value in study_ids])
    if predictions[identifier].duplicated().any():
        raise ValueError("Prediction study IDs are duplicated")
    ordered = sample_submission[[identifier]].astype({identifier: str}).merge(
        predictions, on=identifier, how="left", validate="one_to_one", sort=False
    )
    validate_submission(ordered, sample_submission, identifier=identifier)
    return ordered


def write_submission(
    submission: pd.DataFrame,
    sample_submission: pd.DataFrame,
    path: str | Path,
    *,
    identifier: str = "StudyInstanceUID",
) -> Path:
    """Validate and atomically persist a submission CSV."""
    validate_submission(submission, sample_submission, identifier=identifier)
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    submission.to_csv(temporary, index=False)
    temporary.replace(output)
    return output
