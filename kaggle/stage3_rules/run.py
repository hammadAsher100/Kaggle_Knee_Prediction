"""Run the privacy-preserving report audit and rules_v1 labeler on Kaggle."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

INPUT_ROOT = Path("/kaggle/input")
OUTPUT_ROOT = Path("/kaggle/working/stage3_artifacts")


def discover_repository() -> Path:
    datasets_group = INPUT_ROOT / "datasets"
    candidates = list(INPUT_ROOT.iterdir())
    if datasets_group.is_dir():
        candidates.extend(datasets_group.glob("*/*"))
    matches = sorted(
        path
        for path in candidates
        if path.is_dir()
        and (path / "pyproject.toml").is_file()
        and (path / "scripts" / "build_labels.py").is_file()
    )
    if len(matches) != 1:
        raise RuntimeError(f"Expected one repository source, found {len(matches)}")
    return matches[0]


def run(repository: Path, *arguments: str) -> None:
    command = [sys.executable, *arguments]
    subprocess.run(
        command,
        cwd=repository,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        check=True,
    )


def main() -> int:
    repository = discover_repository()
    train_csv = INPUT_ROOT / "competitions/rsna-knee-abnormality-detection/train.csv"
    if not train_csv.is_file():
        raise FileNotFoundError(f"Training CSV is unavailable: {train_csv}")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    run(
        repository,
        "scripts/audit_report_corpus.py",
        "--train-csv",
        str(train_csv),
        "--output",
        str(OUTPUT_ROOT / "report_corpus_audit.json"),
    )
    run(
        repository,
        "scripts/build_labels.py",
        "--train-csv",
        str(train_csv),
        "--config",
        "configs/labeler.yaml",
        "--output",
        str(OUTPUT_ROOT / "report_labels_rules_v1.parquet"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
