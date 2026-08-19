"""Tests for deterministic provenance hashes."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.utils.hashing import config_hash, file_hash


def test_config_hash_is_order_independent_and_sensitive_to_values() -> None:
    first = {"model": {"size": 224, "name": "tiny"}, "seed": 7}
    reordered = {"seed": 7, "model": {"name": "tiny", "size": 224}}
    changed = {"seed": 8, "model": {"name": "tiny", "size": 224}}

    assert config_hash(first) == config_hash(reordered)
    assert config_hash(first) != config_hash(changed)
    assert len(config_hash(first, length=12)) == 12


def test_config_hash_rejects_nan() -> None:
    with pytest.raises(ValueError):
        config_hash({"bad": float("nan")})


def test_file_hash_streams_content(tmp_path: Path) -> None:
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"rsna-knee")

    assert file_hash(path) == "a30ad6a58bef1002cab70408fd60cac39b2380df715454c53b739cd388e45162"
