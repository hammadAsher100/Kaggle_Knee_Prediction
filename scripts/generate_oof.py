"""Assemble fold predictions into one complete deterministic OOF table."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--training-table", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-splits", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.input_dir)
    paths = [root / f"fold-{fold}-oof.parquet" for fold in range(args.n_splits)]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing fold OOF files: " + ", ".join(missing))
    oof = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    table = pd.read_parquet(args.training_table)
    if oof["StudyInstanceUID"].duplicated().any():
        raise ValueError("OOF contains duplicate study IDs")
    expected = set(table["StudyInstanceUID"].astype(str))
    observed = set(oof["StudyInstanceUID"].astype(str))
    if expected != observed:
        raise ValueError(
            "OOF study mismatch: "
            f"missing={len(expected - observed)}, extra={len(observed - expected)}"
        )
    metadata = table[["StudyInstanceUID", "fold", "PatientSex", "leakage_group"]]
    oof = metadata.merge(oof, on="StudyInstanceUID", validate="one_to_one", sort=False)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    oof.to_parquet(temporary, index=False)
    temporary.replace(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
