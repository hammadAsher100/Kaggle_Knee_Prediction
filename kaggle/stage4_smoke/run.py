"""Exercise real DICOM decoding and DINOv2 extraction on example test studies."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import yaml

INPUT_ROOT = Path("/kaggle/input")


def _mounts(group: str) -> list[Path]:
    values = list(INPUT_ROOT.iterdir())
    grouped = INPUT_ROOT / group
    if grouped.is_dir():
        values.extend(path for path in grouped.iterdir() if path.is_dir())
        if group == "datasets":
            values.extend(grouped.glob("*/*"))
    return values


def _one(candidates: list[Path], description: str) -> Path:
    unique = sorted(set(path.resolve() for path in candidates))
    if len(unique) != 1:
        raise RuntimeError(f"Expected one {description}, found {len(unique)}")
    return unique[0]


def run(repository: Path, *arguments: str) -> None:
    subprocess.run(
        [sys.executable, *arguments],
        cwd=repository,
        env={
            **os.environ,
            "PYTHONDONTWRITEBYTECODE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_OFFLINE": "1",
        },
        check=True,
    )


def main() -> int:
    repository = _one(
        [
            path
            for path in _mounts("datasets")
            if (path / "pyproject.toml").is_file() and (path / "src").is_dir()
        ],
        "repository source",
    )
    competition = _one(
        [
            path
            for path in _mounts("competitions")
            if (path / "test.csv").is_file() and (path / "test_series").is_dir()
        ],
        "competition mount",
    )
    model_path = _one(
        [
            path.parent
            for path in (INPUT_ROOT / "models").rglob("config.json")
            if (path.parent / "pytorch_model.bin").is_file()
        ],
        "DINOv2 model",
    )
    config = yaml.safe_load((repository / "configs" / "frozen_dinov2.yaml").read_text())
    features = config["features"]
    output = Path("/kaggle/working/stage4_smoke")
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "test_series_manifest.parquet"
    run(
        repository,
        "scripts/build_inference_manifest.py",
        "--series-csv",
        str(competition / "test_series.csv"),
        "--dicom-root",
        str(competition / "test_series"),
        "--output",
        str(manifest),
        "--audit",
        str(output / "manifest_audit.json"),
    )
    run(
        repository,
        "scripts/extract_dinov2_features.py",
        "--study-table",
        str(competition / "test.csv"),
        "--series-manifest",
        str(manifest),
        "--dicom-root",
        str(competition / "test_series"),
        "--model-path",
        str(model_path),
        "--output-dir",
        str(output / "features"),
        "--slices-per-series",
        str(features["slices_per_series"]),
        "--max-series",
        str(features["max_series"]),
        "--image-size",
        str(features["image_size"]),
        "--studies-per-part",
        str(features["studies_per_part"]),
        "--num-workers",
        "2",
        "--minimum-valid-stack-fraction",
        str(features["minimum_valid_stack_fraction"]),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
