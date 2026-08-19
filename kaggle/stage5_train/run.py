"""Build grouped folds, train five feature heads, and evaluate OOF predictions."""

from __future__ import annotations

import json
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


def _artifact_roots() -> list[Path]:
    names = (
        "rsna-knee-stage2-aggregate-artifacts",
        "rsna-knee-semantic-labels",
        "rsna-knee-stage4-features",
    )
    roots = [INPUT_ROOT / name for name in names if (INPUT_ROOT / name).is_dir()]
    datasets = INPUT_ROOT / "datasets"
    if datasets.is_dir():
        roots.extend(
            path
            for path in datasets.glob("*/*")
            if path.is_dir() and path.name in names
        )
    return roots


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
    artifact_roots = [INPUT_ROOT / "kernels", *(_artifact_roots())]
    inventory = _one(
        [path for root in artifact_roots if root.is_dir() for path in root.rglob("train_study_inventory.parquet")],
        "study inventory",
    )
    semantic = _one(
        [path for root in artifact_roots if root.is_dir() for path in root.rglob("report_labels_semantic_v1.parquet")],
        "semantic labels",
    )
    features = _one(
        [path for root in artifact_roots if root.is_dir() for path in root.rglob("features.npz")],
        "DINOv2 features",
    )
    config = yaml.safe_load((repository / "configs" / "frozen_dinov2.yaml").read_text())
    training = config["training"]
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
    experiments_dir = output / "experiments"
    for experiment in training["experiments"]:
        experiment_output = experiments_dir / str(experiment["id"])
        run(
            repository,
            "scripts/train_cv.py",
            "--training-table",
            str(training_table),
            "--features",
            str(features),
            "--output-dir",
            str(experiment_output),
            "--n-splits",
            str(training["n_splits"]),
            "--epochs",
            str(training["epochs"]),
            "--batch-size",
            str(training["batch_size"]),
            "--learning-rate",
            str(training["learning_rate"]),
            "--weight-decay",
            str(training["weight_decay"]),
            "--gold-weight",
            str(experiment["gold_weight"]),
            "--patience",
            str(training["patience"]),
            "--dropout",
            str(experiment["dropout"]),
            "--attention-hidden-dim",
            str(experiment["attention_hidden_dim"]),
            "--seed",
            str(experiment["seed"]),
        )
        run(
            repository,
            "scripts/generate_oof.py",
            "--input-dir",
            str(experiment_output),
            "--training-table",
            str(training_table),
            "--output",
            str(experiment_output / "oof.parquet"),
        )
        run(
            repository,
            "scripts/evaluate_oof.py",
            "--oof",
            str(experiment_output / "oof.parquet"),
            "--output",
            str(experiment_output / "oof_metrics.json"),
        )
        metrics = json.loads((experiment_output / "oof_metrics.json").read_text())
        if metrics.get("macro_auc") is not None and float(metrics["macro_auc"]) >= float(
            training["stop_macro_auc"]
        ):
            break
    run(
        repository,
        "scripts/select_best_experiment.py",
        "--experiments-dir",
        str(experiments_dir),
        "--output-dir",
        str(output),
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
    run(
        repository,
        "scripts/profile_model.py",
        "--checkpoint",
        str(output / "fold-0.pt"),
        "--output",
        str(output / "model_profile.json"),
        "--slices",
        str(config["features"]["max_series"] * config["features"]["slices_per_series"]),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
