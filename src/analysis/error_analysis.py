"""Rank-focused OOF failure analysis."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd


def ranked_error_cases(
    frame: pd.DataFrame,
    target_names: Sequence[str],
    *,
    top_k: int = 3,
    identifier: str = "StudyInstanceUID",
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Return highest-ranked negatives and lowest-ranked positives per target."""
    if top_k < 1:
        raise ValueError("top_k must be positive")
    required = {identifier}
    for target in target_names:
        required.update({f"{target}__gold", f"{target}__prediction"})
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"OOF frame is missing columns: {sorted(missing)}")

    results: dict[str, dict[str, list[dict[str, Any]]]] = {}
    detail_columns = [identifier]
    if "fold" in frame:
        detail_columns.append("fold")
    for target in target_names:
        gold_column = f"{target}__gold"
        prediction_column = f"{target}__prediction"
        known = frame.loc[frame[gold_column].notna()].copy()
        if not known[prediction_column].between(0, 1).all():
            raise ValueError(f"predictions are invalid for {target}")

        def records(
            rows: pd.DataFrame,
            score_column: str = prediction_column,
        ) -> list[dict[str, Any]]:
            values: list[dict[str, Any]] = []
            for _, row in rows.iterrows():
                item = {column: row[column] for column in detail_columns}
                item["prediction"] = float(row[score_column])
                values.append(item)
            return values

        negatives = known.loc[known[gold_column].eq(0)].nlargest(top_k, prediction_column)
        positives = known.loc[known[gold_column].eq(1)].nsmallest(top_k, prediction_column)
        results[target] = {
            "highest_ranked_negatives": records(negatives),
            "lowest_ranked_positives": records(positives),
        }
    return results
