"""Build grouped folds, train five feature heads, and evaluate OOF predictions."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

INPUT_ROOT = Path("/kaggle/input")


def _one(candidates: list[Path], description: str) -> Path:
    unique = sorted(set(path.resolve() for path in candidates))
    if len(unique) != 1:
        raise RuntimeError(f"Expected one {description}, found {len(unique)}")
    return unique[0]


def run(repository: Path, *arguments: str) -> None:
    subprocess.run(
        [sys.executable, *arguments],
        cwd=repository,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        check=True,
    )


def _mounts(group: str) -> list[Path]:
    values = list(INPUT_ROOT.iterdir())
    grouped = INPUT_ROOT / group
    if grouped.is_dir():
        values.extend(path for path in grouped.iterdir() if path.is_dir())
        if group == "datasets":
            values.extend(grouped.glob("*/*"))
    return values


def main() -> int:
    repository = _one(
        [
            path
            for path in _mounts("datasets")
            if (path / "pyproject.toml").is_file() and (path / "src").is_dir()
        ],
        "repository source",
    )
    train_csv = _one(
        [
            path / "train.csv"
            for path in _mounts("competitions")
            if (path / "train.csv").is_file() and (path / "train_series").is_dir()
        ],
        "competition train.csv",
    )
    inventory = _one(
        list((INPUT_ROOT / "kernels").rglob("train_study_inventory.parquet")),
        "study inventory",
    )
    semantic = _one(
        list((INPUT_ROOT / "kernels").rglob("report_labels_semantic_v1.parquet")),
        "semantic labels",
    )
    features = _one(
        list((INPUT_ROOT / "kernels").rglob("features.npz")),
        "DINOv2 features",
    )
    output = Path("/kaggle/working/stage5_cv")
    output.mkdir(parents=True, exist_ok=True)
    training_table = output / "training_table.parquet"
    run(
        repository,
        "scripts/build_training_table.py",
        "--train-csv",
        str(train_csv),
        "--semantic-labels",
        str(semantic),
        "--study-inventory",
        str(inventory),
        "--output",
        str(training_table),
        "--audit",
        str(output / "fold_audit.json"),
    )
    run(
        repository,
        "scripts/train_cv.py",
        "--training-table",
        str(training_table),
        "--features",
        str(features),
        "--output-dir",
        str(output),
        "--n-splits",
        "5",
        "--epochs",
        "40",
        "--batch-size",
        "64",
    )
    run(
        repository,
        "scripts/generate_oof.py",
        "--input-dir",
        str(output),
        "--training-table",
        str(training_table),
        "--output",
        str(output / "oof.parquet"),
    )
    run(
        repository,
        "scripts/evaluate_oof.py",
        "--oof",
        str(output / "oof.parquet"),
        "--output",
        str(output / "oof_metrics.json"),
    )
    run(
        repository,
        "scripts/calibrate_oof.py",
        "--oof",
        str(output / "oof.parquet"),
        "--output-oof",
        str(output / "oof_calibrated.parquet"),
        "--parameters",
        str(output / "calibration.json"),
        "--audit",
        str(output / "calibration_audit.json"),
    )
    run(
        repository,
        "scripts/analyze_oof.py",
        "--oof",
        str(output / "oof.parquet"),
        "--output",
        str(output / "oof_analysis.json"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
