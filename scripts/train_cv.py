"""Run all frozen-feature cross-validation folds reproducibly."""

from __future__ import annotations

import argparse
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-table", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gold-weight", type=float, default=4.0)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260812)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for fold in range(args.n_splits):
        command = [
            sys.executable,
            "scripts/train_fold.py",
            "--training-table",
            args.training_table,
            "--features",
            args.features,
            "--output-dir",
            args.output_dir,
            "--fold",
            str(fold),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--learning-rate",
            str(args.learning_rate),
            "--weight-decay",
            str(args.weight_decay),
            "--gold-weight",
            str(args.gold_weight),
            "--patience",
            str(args.patience),
            "--seed",
            str(args.seed),
        ]
        subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
