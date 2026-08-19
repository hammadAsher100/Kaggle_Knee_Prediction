"""Stable hashing helpers for configuration and artifact provenance."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any


def _canonicalize(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _canonicalize(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        sorted_items = sorted(value.items(), key=lambda item: str(item[0]))
        return {str(key): _canonicalize(item) for key, item in sorted_items}
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]
    if isinstance(value, set):
        canonical_items = [_canonicalize(item) for item in value]
        return sorted(canonical_items, key=lambda item: json.dumps(item, sort_keys=True))
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Enum):
        return _canonicalize(value.value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported value in canonical configuration: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    """Serialize supported values deterministically for provenance."""
    return json.dumps(
        _canonicalize(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def config_hash(config: Mapping[str, Any], *, length: int | None = None) -> str:
    """Return a SHA-256 hash of a configuration mapping."""
    digest = hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()
    if length is None:
        return digest
    if not 1 <= length <= len(digest):
        raise ValueError(f"length must be between 1 and {len(digest)}")
    return digest[:length]


def file_hash(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Stream a file and return its SHA-256 digest."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()
