"""Fault-tolerant, pixel-free DICOM metadata extraction and inventories."""

from __future__ import annotations

import json
import logging
import math
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pydicom
from pydicom.errors import InvalidDicomError

DEFAULT_DICOM_TAGS: tuple[str, ...] = (
    "PatientID",
    "AccessionNumber",
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "SOPInstanceUID",
    "SeriesDescription",
    "ProtocolName",
    "SequenceName",
    "ImageOrientationPatient",
    "ImagePositionPatient",
    "InstanceNumber",
    "PixelSpacing",
    "SliceThickness",
    "SpacingBetweenSlices",
    "Rows",
    "Columns",
    "PatientPosition",
    "Laterality",
    "ImageLaterality",
    "Manufacturer",
    "ManufacturerModelName",
    "MagneticFieldStrength",
    "EchoTime",
    "RepetitionTime",
    "PhotometricInterpretation",
    "BitsAllocated",
    "BitsStored",
    "PixelRepresentation",
    "RescaleSlope",
    "RescaleIntercept",
    "PixelPaddingValue",
)


@dataclass(frozen=True)
class MetadataExtractionResult:
    """Summary returned after writing a DICOM metadata table."""

    discovered_files: int
    readable_dicoms: int
    failed_files: int
    output_path: str
    failure_path: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def discover_files(root: str | Path) -> list[Path]:
    """Return every regular file beneath root in stable relative-path order."""
    resolved = Path(root).expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"DICOM root does not exist or is not a directory: {resolved}")
    return sorted(
        (path for path in resolved.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(resolved).as_posix(),
    )


def _to_python(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, Iterable):
        return [_to_python(item) for item in value]
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return str(value)
    return int(converted) if converted.is_integer() else converted


def spatial_slice_coordinate(
    orientation: Sequence[float] | None,
    position: Sequence[float] | None,
) -> float | None:
    """Project position onto the slice normal derived from DICOM orientation."""
    if orientation is None or position is None or len(orientation) != 6 or len(position) != 3:
        return None
    try:
        row = [float(value) for value in orientation[:3]]
        column = [float(value) for value in orientation[3:]]
        point = [float(value) for value in position]
    except (TypeError, ValueError):
        return None
    normal = (
        row[1] * column[2] - row[2] * column[1],
        row[2] * column[0] - row[0] * column[2],
        row[0] * column[1] - row[1] * column[0],
    )
    coordinate = sum(normal[index] * point[index] for index in range(3))
    return coordinate if math.isfinite(coordinate) else None


def read_dicom_metadata(
    path: str | Path,
    *,
    root: str | Path | None = None,
    tags: Sequence[str] = DEFAULT_DICOM_TAGS,
) -> dict[str, Any]:
    """Read selected metadata without decoding pixel data."""
    resolved = Path(path).expanduser().resolve()
    base = Path(root).expanduser().resolve() if root is not None else resolved.parent
    dataset = pydicom.dcmread(
        resolved,
        stop_before_pixels=True,
        specific_tags=list(tags),
        force=False,
    )
    record: dict[str, Any] = {
        "dicom_path": resolved.relative_to(base).as_posix(),
        "file_size_bytes": resolved.stat().st_size,
    }
    for tag in tags:
        record[tag] = _to_python(getattr(dataset, tag, None))
    record["SpatialSliceCoordinate"] = spatial_slice_coordinate(
        record.get("ImageOrientationPatient"),
        record.get("ImagePositionPatient"),
    )
    return record


def extract_metadata(
    root: str | Path,
    output_path: str | Path,
    *,
    tags: Sequence[str] = DEFAULT_DICOM_TAGS,
    failure_path: str | Path | None = None,
    logger: logging.Logger | None = None,
) -> MetadataExtractionResult:
    """Extract all readable DICOM headers, recording every rejected file."""
    source_root = Path(root).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    failures_output = (
        Path(failure_path).expanduser().resolve()
        if failure_path is not None
        else output.with_name(f"{output.stem}_failures.jsonl")
    )
    files = discover_files(source_root)
    records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for path in files:
        try:
            records.append(read_dicom_metadata(path, root=source_root, tags=tags))
        except (InvalidDicomError, OSError, ValueError, KeyError, EOFError, OverflowError) as exc:
            failures.append(
                {
                    "path": path.relative_to(source_root).as_posix(),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame.from_records(records, columns=[
        "dicom_path",
        "file_size_bytes",
        *tags,
        "SpatialSliceCoordinate",
    ])
    temporary_output = output.with_suffix(output.suffix + ".tmp")
    if output.suffix.lower() == ".csv":
        frame.to_csv(temporary_output, index=False)
    elif output.suffix.lower() in {".parquet", ".pq"}:
        frame.to_parquet(temporary_output, index=False)
    else:
        raise ValueError("Metadata output must have a .csv, .parquet, or .pq suffix")
    temporary_output.replace(output)

    failures_output.parent.mkdir(parents=True, exist_ok=True)
    temporary_failures = failures_output.with_suffix(failures_output.suffix + ".tmp")
    with temporary_failures.open("w", encoding="utf-8") as handle:
        for failure in failures:
            handle.write(json.dumps(failure, ensure_ascii=False) + "\n")
    temporary_failures.replace(failures_output)

    if logger is not None:
        logger.info(
            "dicom_metadata_extracted",
            extra={
                "event": "dicom_metadata_extracted",
                "discovered_files": len(files),
                "readable_dicoms": len(records),
                "failed_files": len(failures),
                "output_path": str(output),
            },
        )
    return MetadataExtractionResult(
        discovered_files=len(files),
        readable_dicoms=len(records),
        failed_files=len(failures),
        output_path=str(output),
        failure_path=str(failures_output),
    )


def build_study_inventory(metadata: pd.DataFrame) -> pd.DataFrame:
    """Aggregate series and slice counts plus geometry reliability per study."""
    required = {"StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID"}
    missing = sorted(required.difference(metadata.columns))
    if missing:
        raise ValueError(f"Metadata is missing required columns: {', '.join(missing)}")
    if metadata.empty:
        return pd.DataFrame(
            columns=[
                "StudyInstanceUID",
                "series_count",
                "slice_count",
                "unique_sop_count",
                "geometry_slice_fraction",
            ]
        )
    working = metadata.copy()
    geometry_present = working.get(
        "SpatialSliceCoordinate", pd.Series(index=working.index, dtype=float)
    ).notna()
    working["_geometry_present"] = geometry_present.astype(int)
    inventory = (
        working.groupby("StudyInstanceUID", dropna=False)
        .agg(
            series_count=("SeriesInstanceUID", "nunique"),
            slice_count=("dicom_path", "size"),
            unique_sop_count=("SOPInstanceUID", "nunique"),
            geometry_slice_fraction=("_geometry_present", "mean"),
        )
        .reset_index()
    )
    return inventory


def duplicate_identifier_counts(metadata: pd.DataFrame) -> dict[str, int]:
    """Count duplicated non-null DICOM identifiers."""
    result: dict[str, int] = {}
    for column in ("StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID"):
        if column in metadata:
            values = metadata[column].dropna()
            result[column] = int(values.duplicated(keep=False).sum())
    return result


def missingness(metadata: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    """Return exact missing count and fraction per metadata field."""
    denominator = len(metadata)
    return {
        column: {
            "missing_count": int(metadata[column].isna().sum()),
            "missing_fraction": (
                float(metadata[column].isna().mean()) if denominator else 0.0
            ),
        }
        for column in metadata.columns
    }


def series_slice_distribution(metadata: pd.DataFrame) -> dict[str, Any]:
    """Summarize slice counts per valid series."""
    if metadata.empty or "SeriesInstanceUID" not in metadata:
        return {"series_count": 0, "slice_counts": {}}
    counts = metadata.groupby("SeriesInstanceUID", dropna=False).size()
    return {
        "series_count": int(len(counts)),
        "slice_counts": {
            "min": int(counts.min()),
            "median": float(counts.median()),
            "max": int(counts.max()),
            "frequency": dict(sorted(Counter(counts.astype(int)).items())),
        },
    }


def metadata_audit(metadata: pd.DataFrame) -> dict[str, Any]:
    """Build exact DICOM-level, series-level, and study-level audit statistics."""
    categorical_columns = (
        "SeriesDescription",
        "ProtocolName",
        "SequenceName",
        "PatientPosition",
        "Laterality",
        "ImageLaterality",
        "Manufacturer",
        "ManufacturerModelName",
        "MagneticFieldStrength",
        "PhotometricInterpretation",
        "Rows",
        "Columns",
    )
    distributions: dict[str, dict[str, int]] = {}
    for column in categorical_columns:
        if column in metadata:
            counts = metadata[column].fillna("<missing>").map(str).value_counts(dropna=False)
            distributions[column] = {
                key: int(value) for key, value in counts.items()
            }

    geometry: dict[str, Any] = {
        "series_with_all_coordinates": 0,
        "series_with_unique_coordinates": 0,
        "series_reliably_orderable": 0,
        "series_total": 0,
    }
    if {"SeriesInstanceUID", "SpatialSliceCoordinate"}.issubset(metadata.columns):
        grouped = metadata.groupby("SeriesInstanceUID", dropna=False)["SpatialSliceCoordinate"]
        sizes = grouped.size()
        present = grouped.count()
        unique = grouped.nunique(dropna=True)
        all_present = present == sizes
        all_unique = unique == sizes
        geometry = {
            "series_with_all_coordinates": int(all_present.sum()),
            "series_with_unique_coordinates": int(all_unique.sum()),
            "series_reliably_orderable": int((all_present & all_unique).sum()),
            "series_total": int(len(sizes)),
        }

    hierarchy_depths: dict[str, int] = {}
    if "dicom_path" in metadata:
        depth_counts = metadata["dicom_path"].map(lambda value: len(Path(str(value)).parts))
        hierarchy_depths = {
            str(key): int(value) for key, value in depth_counts.value_counts().sort_index().items()
        }

    return {
        "dicom_count": int(len(metadata)),
        "study_count": (
            int(metadata["StudyInstanceUID"].nunique(dropna=True))
            if "StudyInstanceUID" in metadata
            else None
        ),
        "series_count": (
            int(metadata["SeriesInstanceUID"].nunique(dropna=True))
            if "SeriesInstanceUID" in metadata
            else None
        ),
        "sop_count": (
            int(metadata["SOPInstanceUID"].nunique(dropna=True))
            if "SOPInstanceUID" in metadata
            else None
        ),
        "patient_count_if_recoverable": (
            int(metadata["PatientID"].nunique(dropna=True))
            if "PatientID" in metadata and metadata["PatientID"].notna().any()
            else None
        ),
        "missingness": missingness(metadata),
        "duplicate_identifier_rows": duplicate_identifier_counts(metadata),
        "series_slice_distribution": series_slice_distribution(metadata),
        "geometry": geometry,
        "categorical_distributions": distributions,
        "relative_path_depth_distribution": hierarchy_depths,
    }
