"""Resume Stage 2 from the prior kernel's atomic Parquet parts."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

INPUT_ROOT = Path("/kaggle/input")


def discover_repository() -> Path:
    candidates = [
        path
        for path in INPUT_ROOT.iterdir()
        if (path / "pyproject.toml").is_file() and (path / "src").is_dir()
    ]
    datasets = INPUT_ROOT / "datasets"
    if datasets.is_dir():
        candidates.extend(
            path
            for path in datasets.rglob("*")
            if path.is_dir() and (path / "pyproject.toml").is_file() and (path / "src").is_dir()
        )
    unique = sorted(set(path.resolve() for path in candidates))
    if len(unique) != 1:
        raise RuntimeError(f"Expected one repository source, found {len(unique)}")
    return unique[0]


def discover_prior_manifest() -> Path:
    search_roots = [INPUT_ROOT / "kernels", INPUT_ROOT / "datasets"]
    attached_artifacts = INPUT_ROOT / "rsna-knee-stage2-artifacts"
    if attached_artifacts.is_dir():
        search_roots.append(attached_artifacts)
    manifests = sorted(
        {
            path
            for root in search_roots
            if root.is_dir()
            for path in root.rglob("train_metadata_manifest.json")
            if (path.parent / "train_metadata_parts").is_dir()
        }
    )
    if len(manifests) != 1:
        raise RuntimeError(f"Expected one prior Stage 2 manifest, found {len(manifests)}")
    return manifests[0]


def main() -> int:
    repository = discover_repository()
    prior_manifest = discover_prior_manifest()
    prior_root = prior_manifest.parent
    working_root = Path("/kaggle/working/stage2_artifacts")
    working_root.mkdir(parents=True, exist_ok=True)
    prior_parts = prior_root / "train_metadata_parts"
    target_parts = working_root / "train_metadata_parts"
    if target_parts.exists():
        raise RuntimeError(f"Refusing to overwrite existing resume directory: {target_parts}")
    shutil.copytree(prior_parts, target_parts)
    for name in ("train_metadata_manifest.json", "train_dicom_failures.jsonl"):
        source = prior_root / name
        if source.is_file():
            shutil.copy2(source, working_root / name)
    command = [
        sys.executable,
        "scripts/build_metadata_parts.py",
        "--config",
        "configs/kaggle.yaml",
        "--config",
        "configs/kaggle_stage2.yaml",
        "--split",
        "train",
        "--studies-per-part",
        "25",
        "--resume-manifest",
        str(working_root / "train_metadata_manifest.json"),
    ]
    completed = subprocess.run(
        command,
        cwd=repository,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        check=False,
    )
    return 0 if completed.returncode in {0, 75} else completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
