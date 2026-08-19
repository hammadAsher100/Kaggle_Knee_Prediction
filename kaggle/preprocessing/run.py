"""Bootstrap the tracked repository inside a Kaggle preprocessing kernel."""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile

INPUT_ROOT = Path("/kaggle/input")
WORKSPACE = Path("/kaggle/working/rsna-knee")
REQUIRED_MODULES = ("yaml", "numpy", "pandas", "pyarrow", "pydicom")


def run(*arguments: str) -> None:
    """Run one repository command and stop immediately on failure."""
    command = [sys.executable, *arguments]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=WORKSPACE, check=True)


def discover_source_dataset() -> Path:
    """Locate the attached source snapshot without assuming its mount name."""
    if not INPUT_ROOT.is_dir():
        raise FileNotFoundError(f"Kaggle input root is unavailable: {INPUT_ROOT}")
    candidates = sorted(
        path
        for path in INPUT_ROOT.iterdir()
        if path.is_dir()
        and (path / "pyproject.toml").is_file()
        and (path / "src").is_dir()
        and (path / "configs" / "kaggle.yaml").is_file()
    )
    if len(candidates) != 1:
        mounted = ", ".join(sorted(path.name for path in INPUT_ROOT.iterdir()))
        raise RuntimeError(
            "Expected exactly one attached repository source, found "
            f"{len(candidates)}. Mounted inputs: {mounted}"
        )
    return candidates[0]


def main() -> int:
    source_dataset = discover_source_dataset()

    WORKSPACE.mkdir(parents=True, exist_ok=True)
    if (source_dataset / "pyproject.toml").is_file():
        shutil.copytree(source_dataset, WORKSPACE, dirs_exist_ok=True)
    else:
        archives = sorted(source_dataset.glob("*.zip"))
        if len(archives) != 1:
            raise RuntimeError(
                "Expected an unpacked repository or exactly one archive in "
                f"{source_dataset}, found {len(archives)} archives"
            )
        with ZipFile(archives[0]) as archive:
            archive.extractall(WORKSPACE)

    os.chdir(WORKSPACE)
    sys.path.insert(0, str(WORKSPACE))

    missing = [name for name in REQUIRED_MODULES if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError("Kaggle image is missing required modules: " + ", ".join(missing))

    run("scripts/inspect_environment.py", "--config", "configs/kaggle.yaml")
    run(
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
