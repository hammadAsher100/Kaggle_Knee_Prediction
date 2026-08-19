"""Geometry-derived MRI orientation and slice ordering utilities."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

PLANE_NORMALS: dict[str, np.ndarray] = {
    "Sagittal": np.asarray([1.0, 0.0, 0.0]),
    "Coronal": np.asarray([0.0, 1.0, 0.0]),
    "Axial": np.asarray([0.0, 0.0, 1.0]),
}


def slice_normal(orientation: Sequence[float] | None) -> np.ndarray | None:
    """Return a unit slice normal from DICOM ImageOrientationPatient."""
    if orientation is None or len(orientation) != 6:
        return None
    values = np.asarray(orientation, dtype=float)
    if not np.isfinite(values).all():
        return None
    normal = np.cross(values[:3], values[3:])
    norm = float(np.linalg.norm(normal))
    return normal / norm if norm > 1e-8 else None


def infer_anatomical_plane(
    orientation: Sequence[float] | None,
    *,
    tolerance_degrees: float = 35.0,
) -> str:
    """Classify plane by the nearest patient-axis normal, including oblique scans."""
    normal = slice_normal(orientation)
    if normal is None:
        return "Unknown"
    similarities = {
        plane: float(abs(np.dot(normal, axis))) for plane, axis in PLANE_NORMALS.items()
    }
    plane, similarity = max(similarities.items(), key=lambda item: item[1])
    threshold = float(np.cos(np.deg2rad(tolerance_degrees)))
    return plane if similarity >= threshold else "Oblique"


def spatial_coordinates(
    orientations: Iterable[Sequence[float] | None],
    positions: Iterable[Sequence[float] | None],
) -> list[float | None]:
    """Project positions on each slice normal without relying on filenames."""
    result: list[float | None] = []
    for orientation, position in zip(orientations, positions, strict=True):
        normal = slice_normal(orientation)
        if normal is None or position is None or len(position) != 3:
            result.append(None)
            continue
        point = np.asarray(position, dtype=float)
        result.append(float(np.dot(normal, point)) if np.isfinite(point).all() else None)
    return result


def geometry_order(
    coordinates: Sequence[float | None],
    instance_numbers: Sequence[int | float | None] | None = None,
) -> tuple[np.ndarray, str]:
    """Return a stable slice order and the evidence source used for it."""
    count = len(coordinates)
    coordinate_values = np.asarray(
        [np.nan if value is None else float(value) for value in coordinates], dtype=float
    )
    if (
        count
        and np.isfinite(coordinate_values).all()
        and len(np.unique(coordinate_values)) == count
    ):
        return np.argsort(coordinate_values, kind="stable"), "geometry"
    if instance_numbers is not None and len(instance_numbers) == count:
        instance_values = np.asarray(
            [np.nan if value is None else float(value) for value in instance_numbers], dtype=float
        )
        if (
            count
            and np.isfinite(instance_values).all()
            and len(np.unique(instance_values)) == count
        ):
            return np.argsort(instance_values, kind="stable"), "instance_number"
    return np.arange(count, dtype=int), "input_fallback"
