"""Merge semantic labels, gold labels, metadata inventory, and grouped folds."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.training_table import build_training_table

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
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--semantic-labels", required=True)
    parser.add_argument("--study-inventory", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--restarts", type=int, default=64)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    train = pd.read_csv(args.train_csv)
    semantic = pd.read_parquet(args.semantic_labels)
    inventory = pd.read_parquet(args.study_inventory)
    folded, audit = build_training_table(
        train,
        semantic,
        inventory,
        target_columns=TARGETS,
        n_splits=args.n_splits,
        seed=args.seed,
        restarts=args.restarts,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    folded.to_parquet(temporary, index=False)
    temporary.replace(output)
    audit_output = Path(args.audit)
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    temporary_audit = audit_output.with_suffix(audit_output.suffix + ".tmp")
    temporary_audit.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary_audit.replace(audit_output)
    print(json.dumps({"rows": len(folded), "grouping_source": audit["grouping_source"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
