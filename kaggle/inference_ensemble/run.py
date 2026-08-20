"""Offline feature extraction and equal-probability two-model inference."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import yaml

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
        list(INPUT_ROOT.rglob("pyproject.toml")),
        "repository pyproject",
    ).parent
    test_csv = _one(
        [path for path in INPUT_ROOT.rglob("test.csv") if (path.parent / "test_series").is_dir()],
        "competition test.csv",
    )
    model_path = _one(
        [
            path.parent
            for path in INPUT_ROOT.rglob("config.json")
            if "dinov2" in str(path).lower() and (path.parent / "pytorch_model.bin").is_file()
        ],
        "DINOv2 model",
    )
    checkpoint_dirs = sorted(
        {
            path.parent.resolve()
            for path in INPUT_ROOT.rglob("best_experiment.json")
            if len(list(path.parent.glob("fold-*.pt"))) == 5
        }
    )
    if len(checkpoint_dirs) != 2:
        raise RuntimeError(f"Expected two checkpoint directories, found {len(checkpoint_dirs)}")

    config = yaml.safe_load((repository / "configs" / "frozen_dinov2_strict.yaml").read_text())
    features = config["features"]
    output = Path("/kaggle/working/inference_ensemble")
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "test_series_manifest.parquet"
    run(
        repository,
        "scripts/build_inference_manifest.py",
        "--series-csv",
        str(test_csv.parent / "test_series.csv"),
        "--dicom-root",
        str(test_csv.parent / "test_series"),
        "--output",
        str(manifest),
        "--audit",
        str(output / "test_manifest_audit.json"),
    )
    feature_path = output / "test_features" / "features.npz"
    run(
        repository,
        "scripts/extract_dinov2_features.py",
        "--study-table",
        str(test_csv),
        "--series-manifest",
        str(manifest),
        "--dicom-root",
        str(test_csv.parent / "test_series"),
        "--model-path",
        str(model_path),
        "--output-dir",
        str(feature_path.parent),
        "--slices-per-series",
        str(features["slices_per_series"]),
        "--max-series",
        str(features["max_series"]),
        "--image-size",
        str(features["image_size"]),
        "--studies-per-part",
        str(features["studies_per_part"]),
        "--num-workers",
        str(features["num_workers"]),
        "--minimum-valid-stack-fraction",
        str(features["minimum_valid_stack_fraction"]),
    )
    prediction_paths = []
    for index, checkpoint_dir in enumerate(checkpoint_dirs):
        path = output / f"test_predictions_{index}.parquet"
        run(
            repository,
            "scripts/predict_features.py",
            "--study-table",
            str(test_csv),
            "--features",
            str(feature_path),
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--output",
            str(path),
        )
        prediction_paths.append(path)
    predictions = output / "test_predictions_ensemble.parquet"
    run(
        repository,
        "scripts/ensemble_prediction_files.py",
        "--predictions",
        *(str(path) for path in prediction_paths),
        "--mode",
        "probability",
        "--output",
        str(predictions),
    )
    run(
        repository,
        "scripts/build_submission.py",
        "--predictions",
        str(predictions),
        "--sample-submission",
        str(test_csv.parent / "sample_submission.csv"),
        "--output",
        "/kaggle/working/submission.csv",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
