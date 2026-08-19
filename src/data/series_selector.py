"""Auditable MRI series categorization and selection."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class SeriesSelection:
    series_uid: str
    category: str
    score: float
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def categorize_series(row: Mapping[str, Any]) -> str:
    """Map competition descriptors to a stable sequence/plane category."""
    plane = str(row.get("Anatomical_Plane") or "Unknown").strip().title()
    if plane not in {"Sagittal", "Coronal", "Axial"}:
        plane = "Unknown"
    fluid = row.get("Fluid_Sensitive")
    fat_suppression = row.get("Fat_Suppression")
    contrast = "fluid" if fluid in {1, "1"} else "nonfluid"
    fat = "fs" if fat_suppression in {1, "1"} else "nonfs"
    return f"{plane.lower()}_{contrast}_{fat}"


def rank_series(
    row: Mapping[str, Any],
    *,
    preferred_plane: str | None = None,
) -> tuple[float, tuple[str, ...]]:
    """Score a series from explicit descriptors and basic quality evidence."""
    score = 0.0
    reasons: list[str] = []
    plane = str(row.get("Anatomical_Plane") or "Unknown").title()
    if preferred_plane and plane == preferred_plane.title():
        score += 3.0
        reasons.append("preferred_plane")
    if row.get("Fluid_Sensitive") in {1, "1"}:
        score += 2.0
        reasons.append("fluid_sensitive")
    if row.get("Fat_Suppression") in {1, "1"}:
        score += 1.0
        reasons.append("fat_suppression")
    slices = int(row.get("slice_count") or 0)
    if 12 <= slices <= 96:
        score += 1.0
        reasons.append("typical_slice_count")
    elif slices < 3:
        score -= 5.0
        reasons.append("too_few_slices")
    if bool(row.get("geometry_reliably_orderable", False)):
        score += 1.0
        reasons.append("geometry_orderable")
    return score, tuple(reasons)


def select_series(
    rows: Sequence[Mapping[str, Any]],
    *,
    max_series: int = 3,
    preferred_planes: Sequence[str] = ("Sagittal", "Coronal", "Axial"),
) -> list[SeriesSelection]:
    """Select at most one best series per plane, then fill remaining slots."""
    if max_series < 1:
        raise ValueError("max_series must be positive")
    candidates: list[SeriesSelection] = []
    for row in rows:
        uid = str(row.get("SeriesInstanceUID") or "")
        if not uid:
            raise ValueError("Every series requires SeriesInstanceUID")
        plane = str(row.get("Anatomical_Plane") or "Unknown").title()
        score, reasons = rank_series(
            row,
            preferred_plane=plane if plane in preferred_planes else None,
        )
        candidates.append(SeriesSelection(uid, categorize_series(row), score, reasons))
    ordered = sorted(candidates, key=lambda item: (-item.score, item.series_uid))
    selected: list[SeriesSelection] = []
    used_planes: set[str] = set()
    by_uid = {str(row["SeriesInstanceUID"]): row for row in rows}
    for preferred in preferred_planes:
        match = next(
            (
                item
                for item in ordered
                if str(by_uid[item.series_uid].get("Anatomical_Plane") or "").title() == preferred
                and item.series_uid not in {value.series_uid for value in selected}
            ),
            None,
        )
        if match is not None and len(selected) < max_series:
            selected.append(match)
            used_planes.add(preferred)
    for item in ordered:
        if len(selected) >= max_series:
            break
        if item.series_uid not in {value.series_uid for value in selected}:
            selected.append(item)
    return selected
