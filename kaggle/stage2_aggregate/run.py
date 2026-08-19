"""Aggregate the completed Stage 2 metadata scan on Kaggle."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

INPUT_ROOT = Path("/kaggle/input")


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
        and (path / "scripts" / "aggregate_metadata.py").is_file()
    )
    if len(matches) != 1:
        raise RuntimeError(f"Expected one repository source, found {len(matches)}")
    return matches[0]


def discover_completed_parts() -> Path:
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
        }
    )
    completed = []
    for manifest in manifests:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        parts = manifest.parent / "train_metadata_parts"
        if payload.get("complete") is True and parts.is_dir():
            completed.append(parts)
    if len(completed) != 1:
        raise RuntimeError(
            f"Expected one completed Stage 2 parts directory, found {len(completed)}"
        )
    return completed[0]


def main() -> int:
    repository = discover_repository()
    parts = discover_completed_parts()
    command = [
        sys.executable,
        "scripts/aggregate_metadata.py",
        "--config",
        "configs/kaggle.yaml",
        "--config",
        "configs/kaggle_stage2.yaml",
        "--parts-dir",
        str(parts),
    ]
    completed = subprocess.run(
        command,
        cwd=repository,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        check=False,
    )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
