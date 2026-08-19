"""Bootstrap the tracked repository inside a Kaggle preprocessing kernel."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

INPUT_ROOT = Path("/kaggle/input")
REQUIRED_MODULES = ("yaml", "numpy", "pandas", "pyarrow", "pydicom")


def run(repository: Path, *arguments: str) -> None:
    """Run one repository command and stop immediately on failure."""
    command = [sys.executable, *arguments]
    print("+", " ".join(command), flush=True)
    subprocess.run(
        command,
        cwd=repository,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        check=True,
    )


def discover_source_dataset() -> Path:
    """Locate the attached source snapshot without assuming its mount name."""
    if not INPUT_ROOT.is_dir():
        raise FileNotFoundError(f"Kaggle input root is unavailable: {INPUT_ROOT}")
    possible_mounts = list(INPUT_ROOT.iterdir())
    datasets_group = INPUT_ROOT / "datasets"
    if datasets_group.is_dir():
        possible_mounts.extend(datasets_group.glob("*/*"))
    candidates = sorted(
        path
        for path in possible_mounts
        if path.is_dir()
        and (path / "pyproject.toml").is_file()
        and (path / "src").is_dir()
        and (path / "configs" / "kaggle.yaml").is_file()
    )
    if len(candidates) != 1:
        mounted = ", ".join(
            sorted(path.relative_to(INPUT_ROOT).as_posix() for path in possible_mounts)
        )
        raise RuntimeError(
            "Expected exactly one attached repository source, found "
            f"{len(candidates)}. Mounted inputs: {mounted}"
        )
    return candidates[0]


def main() -> int:
    source_dataset = discover_source_dataset()

    missing = [name for name in REQUIRED_MODULES if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError("Kaggle image is missing required modules: " + ", ".join(missing))

    run(
        source_dataset,
        "scripts/inspect_environment.py",
        "--config",
        "configs/kaggle.yaml",
    )
    run(
        source_dataset,
        "scripts/audit_data.py",
        "--config",
        "configs/kaggle.yaml",
        "--config",
        "configs/kaggle_stage2.yaml",
    )

    print("Kaggle repository bootstrap and table audit completed successfully.", flush=True)
    print("Outputs: /kaggle/working/stage2_artifacts", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
