"""Cross-fit OOF calibration and fit final monotonic calibrators for inference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.inference.calibration import apply_platt, fit_monotonic_platt
from src.training.metrics import multilabel_roc_auc

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
    parser.add_argument("--oof", required=True)
    parser.add_argument("--output-oof", required=True)
    parser.add_argument("--parameters", required=True)
    parser.add_argument("--audit", required=True)
    args = parser.parse_args()
    frame = pd.read_parquet(args.oof)
    cross_fitted = np.zeros((len(frame), len(TARGETS)), dtype=float)
    full_parameters: dict[str, dict[str, object]] = {}
    for target_index, target in enumerate(TARGETS):
        truth = frame[f"{target}__gold"].to_numpy(float)
        raw = frame[f"{target}__prediction"].to_numpy(float)
        for fold in sorted(frame["fold"].unique()):
            validation = frame["fold"].eq(fold).to_numpy()
            parameters = fit_monotonic_platt(truth[~validation], raw[~validation])
            cross_fitted[validation, target_index] = apply_platt(raw[validation], parameters)
        full_parameters[target] = fit_monotonic_platt(truth, raw)
        frame[f"{target}__calibrated_prediction"] = cross_fitted[:, target_index]
    truth_matrix = frame[[f"{target}__gold" for target in TARGETS]].to_numpy(float)
    raw_matrix = frame[[f"{target}__prediction" for target in TARGETS]].to_numpy(float)
    raw_metric = multilabel_roc_auc(truth_matrix, raw_matrix, TARGETS)
    calibrated_metric = multilabel_roc_auc(truth_matrix, cross_fitted, TARGETS)
    audit = {
        "method": "positive_slope_platt",
        "cross_fitted": True,
        "raw_gold_auc": raw_metric.to_dict(),
        "calibrated_gold_auc": calibrated_metric.to_dict(),
        "full_parameters": full_parameters,
    }
    output_oof = Path(args.output_oof)
    temporary_oof = output_oof.with_suffix(output_oof.suffix + ".tmp")
    frame.to_parquet(temporary_oof, index=False)
    temporary_oof.replace(output_oof)
    Path(args.parameters).write_text(
        json.dumps(full_parameters, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    Path(args.audit).write_text(
        json.dumps(audit, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
