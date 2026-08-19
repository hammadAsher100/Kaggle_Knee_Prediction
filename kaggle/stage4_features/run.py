"""Extract frozen DINOv2-small features for all training studies."""

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
    series_manifest = _one(
        list((INPUT_ROOT / "kernels").rglob("train_series_manifest.parquet")),
        "aggregated series manifest",
    )
    train_csv = _one(
        [
            path / "train.csv"
            for path in _mounts("competitions")
            if (path / "train.csv").is_file() and (path / "train_series").is_dir()
        ],
        "competition train.csv",
    )
    model_path = _one(
        [
            path.parent
            for path in (INPUT_ROOT / "models").rglob("config.json")
            if (path.parent / "pytorch_model.bin").is_file()
        ],
        "DINOv2 model",
    )
    command = [
        sys.executable,
        "scripts/extract_dinov2_features.py",
        "--study-table",
        str(train_csv),
        "--series-manifest",
        str(series_manifest),
        "--dicom-root",
        str(train_csv.parent / "train_series"),
        "--model-path",
        str(model_path),
        "--output-dir",
        "/kaggle/working/stage4_features",
        "--slices-per-series",
        "4",
        "--max-series",
        "3",
        "--studies-per-part",
        "50",
        "--num-workers",
        "4",
    ]
    completed = subprocess.run(
        command,
        cwd=repository,
        env={
            **os.environ,
            "PYTHONDONTWRITEBYTECODE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_OFFLINE": "1",
        },
        check=False,
    )
    return 0 if completed.returncode == 75 else completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
