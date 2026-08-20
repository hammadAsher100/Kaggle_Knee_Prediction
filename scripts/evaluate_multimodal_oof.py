from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.multimodal_fusion import nested_simplex_fusion

DEFAULT_TARGETS = [
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-oof", type=Path, required=True)
    parser.add_argument("--report-labels", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-oof", type=Path)
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument(
        "--per-target",
        action="store_true",
        help="Select separate weights per target; less stable with very sparse gold labels.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image = pd.read_parquet(args.image_oof)
    report = pd.read_parquet(args.report_labels)
    frame = image.merge(
        report,
        on="StudyInstanceUID",
        how="inner",
        validate="one_to_one",
        suffixes=("", "__report"),
    )
    if len(frame) != len(image) or len(frame) != len(report):
        raise ValueError("image and report artifacts do not contain identical study IDs")

    y_true = frame[[f"{target}__gold" for target in DEFAULT_TARGETS]].to_numpy(float)
    modality_arrays = [
        frame[[f"{target}__prediction" for target in DEFAULT_TARGETS]].to_numpy(float),
        frame[[f"{target}__semantic_probability" for target in DEFAULT_TARGETS]].to_numpy(float),
        frame[[f"{target}__rule_probability" for target in DEFAULT_TARGETS]].to_numpy(float),
    ]
    result = nested_simplex_fusion(
        y_true,
        np.stack(modality_arrays, axis=2),
        frame["fold"].to_numpy(),
        step=args.step,
        per_target=args.per_target,
    )

    payload = {
        "evaluation": "nested_oof_simplex_fusion",
        "macro_auc": result.macro_auc,
        "modalities": ["image", "semantic_report", "rule_report"],
        "per_target": dict(zip(DEFAULT_TARGETS, result.per_target_auc.tolist(), strict=True)),
        "per_target_weights": args.per_target,
        "step": args.step,
        "fold_weights": {str(key): value.tolist() for key, value in result.weights.items()},
        "gold_rows": int(np.isfinite(y_true).all(axis=1).sum()),
        "limitations": [
            "Only 58 studies have gold labels, so the estimate has high variance.",
            "Weights are selected without each evaluated outer fold, preventing direct "
            "label leakage.",
            "This is a local cross-validation result, not a Kaggle leaderboard score.",
        ],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if args.output_oof:
        output = frame[["StudyInstanceUID", "fold"]].copy()
        for index, target in enumerate(DEFAULT_TARGETS):
            output[f"{target}__prediction"] = result.predictions[:, index]
            output[f"{target}__gold"] = y_true[:, index]
        args.output_oof.parent.mkdir(parents=True, exist_ok=True)
        output.to_parquet(args.output_oof, index=False)

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
