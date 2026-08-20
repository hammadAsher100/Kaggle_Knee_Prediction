"""Align and ensemble multiple study-level prediction Parquet files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.inference.ensemble import weighted_probability_average, weighted_rank_average


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("probability", "rank"), default="probability")
    args = parser.parse_args()
    if len(args.predictions) < 2:
        raise ValueError("At least two prediction files are required")

    identifier = "StudyInstanceUID"
    frames = [pd.read_parquet(path) for path in args.predictions]
    prediction_columns = [column for column in frames[0] if column.endswith("__prediction")]
    if not prediction_columns:
        raise ValueError("Reference file contains no prediction columns")
    reference_ids = frames[0][identifier].astype(str)
    if reference_ids.duplicated().any():
        raise ValueError("Reference prediction IDs are duplicated")

    arrays = []
    for frame in frames:
        if set(frame.columns) != {identifier, *prediction_columns}:
            raise ValueError("Prediction schemas differ")
        candidate_ids = frame[identifier].astype(str)
        if candidate_ids.duplicated().any() or set(candidate_ids) != set(reference_ids):
            raise ValueError("Prediction study IDs differ or are duplicated")
        aligned = pd.DataFrame({identifier: reference_ids}).merge(
            frame.assign(**{identifier: candidate_ids}),
            on=identifier,
            how="left",
            validate="one_to_one",
            sort=False,
        )
        arrays.append(aligned[prediction_columns].to_numpy(float))

    values = (
        weighted_probability_average(arrays)
        if args.mode == "probability"
        else weighted_rank_average(arrays)
    )
    output = pd.DataFrame(values, columns=prediction_columns)
    output.insert(0, identifier, reference_ids)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    output.to_parquet(temporary, index=False)
    temporary.replace(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
