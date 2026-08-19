"""Fault-tolerant, pixel-free DICOM metadata extraction and inventories."""

from __future__ import annotations

import json
import logging
import math
import time
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pydicom
from pydicom.errors import InvalidDicomError

from src.utils.time_guard import TimeGuard

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


@dataclass(frozen=True)
class StreamingMetadataResult:
    """Summary of a resumable, study-partitioned metadata extraction run."""

    requested_studies: int
    completed_studies: int
    remaining_studies: int
    discovered_files: int
    readable_dicoms: int
    failed_files: int
    part_count: int
    complete: bool
    output_dir: str
    manifest_path: str
    failure_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


STRING_METADATA_TAGS = {
    "PatientID",
    "AccessionNumber",
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "SOPInstanceUID",
    "SeriesDescription",
    "ProtocolName",
    "SequenceName",
    "PatientPosition",
    "Laterality",
    "ImageLaterality",
    "Manufacturer",
    "ManufacturerModelName",
    "PhotometricInterpretation",
}
INTEGER_METADATA_TAGS = {
    "InstanceNumber",
    "Rows",
    "Columns",
    "BitsAllocated",
    "BitsStored",
    "PixelRepresentation",
}
FLOAT_METADATA_TAGS = {
    "SliceThickness",
    "SpacingBetweenSlices",
    "MagneticFieldStrength",
    "EchoTime",
    "RepetitionTime",
    "RescaleSlope",
    "RescaleIntercept",
    "PixelPaddingValue",
}


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


def _storage_schema(tags: Sequence[str]) -> pa.Schema:
    fields = [pa.field("dicom_path", pa.string()), pa.field("file_size_bytes", pa.int64())]
    for tag in tags:
        if tag in INTEGER_METADATA_TAGS:
            data_type = pa.int64()
        elif tag in FLOAT_METADATA_TAGS:
            data_type = pa.float64()
        else:
            data_type = pa.string()
        fields.append(pa.field(tag, data_type))
    fields.append(pa.field("SpatialSliceCoordinate", pa.float64()))
    return pa.schema(fields)


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def _storage_record(record: Mapping[str, Any], tags: Sequence[str]) -> dict[str, Any]:
    stored: dict[str, Any] = {
        "dicom_path": str(record["dicom_path"]),
        "file_size_bytes": int(record["file_size_bytes"]),
    }
    for tag in tags:
        value = record.get(tag)
        if tag in INTEGER_METADATA_TAGS:
            numeric = _finite_float(value)
            stored[tag] = int(numeric) if numeric is not None else None
        elif tag in FLOAT_METADATA_TAGS:
            stored[tag] = _finite_float(value)
        elif value is None:
            stored[tag] = None
        elif isinstance(value, (list, tuple, dict)):
            stored[tag] = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        else:
            stored[tag] = str(value)
    stored["SpatialSliceCoordinate"] = _finite_float(record.get("SpatialSliceCoordinate"))
    return stored


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_completed_studies(resume_manifest: str | Path | None) -> set[str]:
    if resume_manifest is None:
        return set()
    path = Path(resume_manifest).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    studies = payload.get("completed_study_ids", [])
    if not isinstance(studies, list) or not all(isinstance(value, str) for value in studies):
        raise ValueError("Resume manifest completed_study_ids must be a list of strings")
    return set(studies)


def extract_metadata_parts(
    root: str | Path,
    study_ids: Sequence[str],
    output_dir: str | Path,
    *,
    tags: Sequence[str] = DEFAULT_DICOM_TAGS,
    manifest_path: str | Path | None = None,
    failure_path: str | Path | None = None,
    resume_manifest: str | Path | None = None,
    studies_per_part: int = 25,
    max_runtime_seconds: float = 9 * 60 * 60,
    safety_reserve_seconds: float = 20 * 60,
    logger: logging.Logger | None = None,
) -> StreamingMetadataResult:
    """Stream DICOM headers into atomic Parquet parts with study-level checkpoints."""
    if studies_per_part <= 0:
        raise ValueError("studies_per_part must be positive")
    source_root = Path(root).expanduser().resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"DICOM root does not exist or is not a directory: {source_root}")
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    manifest = (
        Path(manifest_path).expanduser().resolve()
        if manifest_path is not None
        else destination.parent / f"{destination.name}_manifest.json"
    )
    failures = (
        Path(failure_path).expanduser().resolve()
        if failure_path is not None
        else destination.parent / f"{destination.name}_failures.jsonl"
    )
    failures.parent.mkdir(parents=True, exist_ok=True)

    requested = sorted({str(value) for value in study_ids if str(value).strip()})
    previously_completed = _read_completed_studies(resume_manifest)
    completed = [study_id for study_id in requested if study_id in previously_completed]
    pending = [study_id for study_id in requested if study_id not in previously_completed]
    existing_parts = sorted(destination.glob("part-*.parquet"))
    next_part_index = len(existing_parts)
    schema = _storage_schema(tags)
    log = logger or logging.getLogger("rsna_knee.metadata")
    guard = TimeGuard(
        max_runtime_seconds=max_runtime_seconds,
        safety_reserve_seconds=safety_reserve_seconds,
        logger=log,
    )

    discovered_files = 0
    readable_dicoms = 0
    failed_files = 0
    buffered_records: list[dict[str, Any]] = []
    buffered_studies = 0

    def write_manifest(*, complete: bool) -> None:
        _atomic_json(
            {
                "requested_studies": len(requested),
                "completed_studies": len(completed),
                "remaining_studies": len(requested) - len(completed),
                "completed_study_ids": completed,
                "discovered_files_this_run": discovered_files,
                "readable_dicoms_this_run": readable_dicoms,
                "failed_files_this_run": failed_files,
                "part_count_this_run": next_part_index,
                "complete": complete,
                "time_guard": guard.state().to_dict(),
            },
            manifest,
        )

    def flush_part() -> None:
        nonlocal buffered_records, buffered_studies, next_part_index
        if not buffered_records:
            return
        part = destination / f"part-{next_part_index:05d}.parquet"
        temporary = part.with_suffix(part.suffix + ".tmp")
        table = pa.Table.from_pylist(buffered_records, schema=schema)
        pq.write_table(table, temporary, compression="zstd", use_dictionary=True)
        temporary.replace(part)
        log.info(
            "metadata_part_written",
            extra={
                "event": "metadata_part_written",
                "part_path": str(part),
                "part_rows": len(buffered_records),
                "completed_studies": len(completed),
            },
        )
        next_part_index += 1
        buffered_records = []
        buffered_studies = 0
        write_manifest(complete=False)

    with failures.open("a", encoding="utf-8") as failure_handle:
        for study_id in pending:
            if guard.should_stop():
                flush_part()
                write_manifest(complete=False)
                break
            operation_started = time.monotonic()
            study_dir = source_root / study_id
            if not study_dir.is_dir():
                failure_handle.write(
                    json.dumps(
                        {
                            "path": study_id,
                            "error_type": "MissingStudyDirectory",
                            "error": f"Study directory does not exist: {study_dir}",
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                failed_files += 1
            else:
                files = sorted(
                    (path for path in study_dir.rglob("*") if path.is_file()),
                    key=lambda path: path.relative_to(source_root).as_posix(),
                )
                discovered_files += len(files)
                for path in files:
                    try:
                        record = read_dicom_metadata(path, root=source_root, tags=tags)
                        buffered_records.append(_storage_record(record, tags))
                        readable_dicoms += 1
                    except (
                        InvalidDicomError,
                        OSError,
                        ValueError,
                        KeyError,
                        EOFError,
                        OverflowError,
                    ) as exc:
                        failure_handle.write(
                            json.dumps(
                                {
                                    "path": path.relative_to(source_root).as_posix(),
                                    "error_type": type(exc).__name__,
                                    "error": str(exc),
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        failed_files += 1
            completed.append(study_id)
            buffered_studies += 1
            guard.record_operation(max(0.0, time.monotonic() - operation_started))
            if buffered_studies >= studies_per_part:
                flush_part()
            if len(completed) % 100 == 0:
                log.info(
                    "metadata_progress",
                    extra={
                        "event": "metadata_progress",
                        "completed_studies": len(completed),
                        "requested_studies": len(requested),
                        "readable_dicoms": readable_dicoms,
                        "failed_files": failed_files,
                        "elapsed_seconds": guard.elapsed_seconds,
                    },
                )
        else:
            flush_part()
            write_manifest(complete=True)

    is_complete = len(completed) == len(requested)
    if not manifest.is_file():
        write_manifest(complete=is_complete)
    return StreamingMetadataResult(
        requested_studies=len(requested),
        completed_studies=len(completed),
        remaining_studies=len(requested) - len(completed),
        discovered_files=discovered_files,
        readable_dicoms=readable_dicoms,
        failed_files=failed_files,
        part_count=next_part_index,
        complete=is_complete,
        output_dir=str(destination),
        manifest_path=str(manifest),
        failure_path=str(failures),
    )


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
