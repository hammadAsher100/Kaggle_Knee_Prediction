"""Bootstrap the tracked repository inside a Kaggle preprocessing kernel."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile

SOURCE_DATASET = Path("/kaggle/input/rsna-knee-source")
WORKSPACE = Path("/kaggle/working/rsna-knee")
REQUIRED_MODULES = ("yaml", "numpy", "pandas", "pyarrow", "pydicom")


def run(*arguments: str) -> None:
    """Run one repository command and stop immediately on failure."""
    command = [sys.executable, *arguments]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=WORKSPACE, check=True)


def main() -> int:
    if not SOURCE_DATASET.is_dir():
        raise FileNotFoundError(
            "The private source dataset is not mounted at " f"{SOURCE_DATASET}"
        )

    archives = sorted(SOURCE_DATASET.glob("*.zip"))
    if len(archives) != 1:
        raise RuntimeError(
            "Expected exactly one repository archive in "
            f"{SOURCE_DATASET}, found {len(archives)}"
        )

    WORKSPACE.mkdir(parents=True, exist_ok=True)
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
