"""Run the resumable full training DICOM metadata scan on Kaggle."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

INPUT_ROOT = Path("/kaggle/input")


def discover_source_dataset() -> Path:
    """Locate the attached repository snapshot in current Kaggle mounts."""
    datasets_group = INPUT_ROOT / "datasets"
    possible_mounts = list(INPUT_ROOT.iterdir())
    if datasets_group.is_dir():
        possible_mounts.extend(datasets_group.glob("*/*"))
    candidates = sorted(
        path
        for path in possible_mounts
        if path.is_dir()
        and (path / "pyproject.toml").is_file()
        and (path / "src" / "data" / "metadata.py").is_file()
    )
    if len(candidates) != 1:
        mounted = ", ".join(
            sorted(path.relative_to(INPUT_ROOT).as_posix() for path in possible_mounts)
        )
        raise RuntimeError(
            f"Expected one repository source, found {len(candidates)}. Inputs: {mounted}"
        )
    return candidates[0]


def main() -> int:
    repository = discover_source_dataset()
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
    ]
    completed = subprocess.run(
        command,
        cwd=repository,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        check=False,
    )
    if completed.returncode not in {0, 75}:
        return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
