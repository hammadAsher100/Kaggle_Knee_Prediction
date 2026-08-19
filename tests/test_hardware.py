"""Tests for lightweight reproducibility metadata."""

from __future__ import annotations

from pathlib import Path

from src.utils.hardware import git_commit, runtime_metadata


def test_runtime_metadata_does_not_require_torch_import() -> None:
    metadata = runtime_metadata(include_torch=False)

    assert "python" in metadata
    assert "platform" in metadata
    assert "packages" in metadata
    assert "torch_runtime" not in metadata


def test_non_git_directory_has_no_commit(tmp_path: Path) -> None:
    assert git_commit(tmp_path) is None
