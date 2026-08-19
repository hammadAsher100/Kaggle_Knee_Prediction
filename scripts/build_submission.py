"""Build a strictly validated competition submission from study predictions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.inference.submission import build_submission, write_submission


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--sample-submission", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    predictions = pd.read_parquet(args.predictions)
    sample = pd.read_csv(args.sample_submission)
    targets = [column for column in sample if column != "StudyInstanceUID"]
    prediction_columns = [f"{target}__prediction" for target in targets]
    submission = build_submission(
        predictions["StudyInstanceUID"].astype(str).tolist(),
        predictions[prediction_columns].to_numpy(float),
        sample,
    )
    write_submission(submission, sample, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
