"""Reproducibility metadata for runtime, packages, Git, and optional accelerators."""

from __future__ import annotations

import importlib.metadata
import platform
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def package_versions(packages: Iterable[str]) -> dict[str, str | None]:
    """Return installed versions without importing heavyweight packages."""
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def git_commit(repository: str | Path = ".") -> str | None:
    """Return the current commit or None for an unborn/non-Git repository."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(repository),
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def runtime_metadata(*, include_torch: bool = False) -> dict[str, Any]:
    """Collect cheap host metadata and optionally query PyTorch/CUDA."""
    metadata: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "packages": package_versions(["numpy", "pandas", "pydicom", "PyYAML", "torch", "timm"]),
    }
    if include_torch:
        try:
            import torch
        except ImportError:
            metadata["torch_runtime"] = None
        else:
            metadata["torch_runtime"] = {
                "version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(),
                "gpu_names": [
                    torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
                ],
            }
    return metadata

