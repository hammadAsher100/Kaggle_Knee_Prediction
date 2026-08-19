"""Evaluate complete OOF predictions on gold labels only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frame = pd.read_parquet(args.oof)
    truth = frame[[f"{target}__gold" for target in TARGETS]].to_numpy(float)
    probabilities = frame[[f"{target}__prediction" for target in TARGETS]].to_numpy(float)
    result = multilabel_roc_auc(truth, probabilities, TARGETS)
    brier: dict[str, float | None] = {}
    for index, target in enumerate(TARGETS):
        mask = np.isfinite(truth[:, index])
        brier[target] = (
            float(np.mean((truth[mask, index] - probabilities[mask, index]) ** 2))
            if mask.any()
            else None
        )
    audit = {
        **result.to_dict(),
        "gold_brier_score": brier,
        "row_count": len(frame),
        "fold_counts": {
            str(key): int(value) for key, value in frame["fold"].value_counts().sort_index().items()
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(audit, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(json.dumps(audit, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
