"""Evaluate fixed equal-weight probability and rank ensembles of image OOF files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.inference.ensemble import weighted_probability_average, weighted_rank_average
from src.training.multimodal_fusion import score_targets

TARGETS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.oof) < 2:
        raise ValueError("At least two OOF files are required")

    frames = [pd.read_parquet(path) for path in args.oof]
    identifier = "StudyInstanceUID"
    reference_ids = frames[0][identifier].astype(str)
    if reference_ids.duplicated().any():
        raise ValueError("Reference OOF contains duplicate study IDs")
    ordered: list[pd.DataFrame] = [frames[0]]
    for frame in frames[1:]:
        candidate_ids = frame[identifier].astype(str)
        if candidate_ids.duplicated().any():
            raise ValueError("OOF contains duplicate study IDs")
        if set(candidate_ids) != set(reference_ids):
            raise ValueError("OOF study IDs differ")
        aligned = pd.DataFrame({identifier: reference_ids}).merge(
            frame.assign(**{identifier: candidate_ids}),
            on=identifier,
            how="left",
            validate="one_to_one",
            sort=False,
        )
        ordered.append(aligned)

    gold_columns = [f"{target}__gold" for target in TARGETS]
    prediction_columns = [f"{target}__prediction" for target in TARGETS]
    truth = ordered[0][gold_columns].to_numpy(float)
    for frame in ordered[1:]:
        candidate_truth = frame[gold_columns].to_numpy(float)
        if not np.allclose(truth, candidate_truth, equal_nan=True):
            raise ValueError("OOF gold labels differ")
    predictions = [frame[prediction_columns].to_numpy(float) for frame in ordered]

    methods = {
        "equal_probability": weighted_probability_average(predictions),
        "equal_rank": weighted_rank_average(predictions),
    }
    payload: dict[str, object] = {"inputs": [str(path) for path in args.oof]}
    for name, values in methods.items():
        macro_auc, per_target = score_targets(truth, values)
        payload[name] = {
            "macro_auc": macro_auc,
            "per_target": dict(zip(TARGETS, per_target.tolist(), strict=True)),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
