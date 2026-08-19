"""Run offline multilingual semantic report labeling on a Kaggle GPU."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

INPUT_ROOT = Path("/kaggle/input")
OUTPUT_ROOT = Path("/kaggle/working/stage3_semantic_artifacts")


def discover_repository() -> Path:
    candidates = list(INPUT_ROOT.iterdir())
    datasets_group = INPUT_ROOT / "datasets"
    if datasets_group.is_dir():
        candidates.extend(datasets_group.glob("*/*"))
    matches = [
        path
        for path in candidates
        if path.is_dir() and (path / "scripts" / "build_semantic_labels.py").is_file()
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one repository source, found {len(matches)}")
    return matches[0]


def discover_model() -> Path:
    model_root = INPUT_ROOT / "models"
    matches = sorted(model_root.rglob("config_sentence_transformers.json"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one sentence-transformer model, found {len(matches)}")
    return matches[0].parent


def main() -> int:
    repository = discover_repository()
    model = discover_model()
    train_csv = INPUT_ROOT / "competitions/rsna-knee-abnormality-detection/train.csv"
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "scripts/build_semantic_labels.py",
        "--train-csv",
        str(train_csv),
        "--model-path",
        str(model),
        "--config",
        "configs/labeler.yaml",
        "--output",
        str(OUTPUT_ROOT / "report_labels_semantic_v1.parquet"),
        "--batch-size",
        "256",
    ]
    completed = subprocess.run(
        command,
        cwd=repository,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "TRANSFORMERS_OFFLINE": "1"},
        check=False,
    )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
