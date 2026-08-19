"""Evidence-based laterality normalization without unsafe guessing."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def normalize_laterality(value: Any) -> str | None:
    """Normalize explicit DICOM laterality values to L/R, else return None."""
    if value is None:
        return None
    text = str(value).strip().upper()
    if text in {"L", "LEFT"}:
        return "L"
    if text in {"R", "RIGHT"}:
        return "R"
    return None


def infer_laterality(metadata: Mapping[str, Any]) -> tuple[str | None, str]:
    """Resolve laterality using explicit tags, recording ambiguity as unknown."""
    evidence = [
        (name, normalize_laterality(metadata.get(name)))
        for name in ("ImageLaterality", "Laterality")
    ]
    observed = {value for _, value in evidence if value is not None}
    if len(observed) == 1:
        value = next(iter(observed))
        source = "+".join(name for name, candidate in evidence if candidate == value)
        return value, source
    if len(observed) > 1:
        return None, "conflicting_tags"
    return None, "unavailable"
