"""Aggregate streamed DICOM metadata parts into study and series manifests."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.metadata import metadata_audit

SERIES_SUMMARY_COLUMNS: tuple[str, ...] = (
    "PatientID",
    "AccessionNumber",
    "SeriesDescription",
    "ProtocolName",
    "SequenceName",
    "PatientPosition",
    "Laterality",
    "ImageLaterality",
    "Manufacturer",
    "ManufacturerModelName",
    "MagneticFieldStrength",
    "EchoTime",
    "RepetitionTime",
    "Rows",
    "Columns",
    "PixelSpacing",
    "SliceThickness",
    "SpacingBetweenSlices",
)


def discover_metadata_parts(root: str | Path) -> list[Path]:
    """Return stable Parquet part paths from a completed extraction directory."""
    resolved = Path(root).expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"Metadata parts directory does not exist: {resolved}")
    parts = sorted(resolved.glob("part-*.parquet"))
    if not parts:
        raise FileNotFoundError(f"No metadata Parquet parts found under {resolved}")
    return parts


def read_metadata_parts(root: str | Path) -> pd.DataFrame:
    """Load all validated metadata parts into one frame for final aggregation."""
    frames = [pd.read_parquet(path) for path in discover_metadata_parts(root)]
    return pd.concat(frames, ignore_index=True)


def _first_non_null(values: Iterable[Any]) -> Any:
    for value in values:
        if pd.notna(value):
            return value
    return None


def _ordered_series_record(group: pd.DataFrame) -> dict[str, Any]:
    coordinates = group["SpatialSliceCoordinate"]
    instances = group["InstanceNumber"]
    geometry_reliable = bool(
        coordinates.notna().all() and coordinates.nunique(dropna=True) == len(group)
    )
    instance_reliable = bool(
        instances.notna().all() and instances.nunique(dropna=True) == len(group)
    )
    if geometry_reliable:
        ordered = group.sort_values(
            ["SpatialSliceCoordinate", "InstanceNumber", "dicom_path"],
            kind="stable",
            na_position="last",
        )
        order_method = "geometry"
    elif instance_reliable:
        ordered = group.sort_values(
            ["InstanceNumber", "dicom_path"],
            kind="stable",
            na_position="last",
        )
        order_method = "instance_number"
    else:
        ordered = group.sort_values("dicom_path", kind="stable")
        order_method = "path_fallback"

    record: dict[str, Any] = {
        "StudyInstanceUID": str(group["StudyInstanceUID"].iloc[0]),
        "SeriesInstanceUID": str(group["SeriesInstanceUID"].iloc[0]),
        "slice_count": int(len(group)),
        "unique_sop_count": int(group["SOPInstanceUID"].nunique(dropna=True)),
        "geometry_slice_fraction": float(coordinates.notna().mean()),
        "instance_number_fraction": float(instances.notna().mean()),
        "geometry_reliably_orderable": geometry_reliable,
        "instance_reliably_orderable": instance_reliable,
        "order_method": order_method,
        "ordered_dicom_paths": ordered["dicom_path"].astype(str).tolist(),
        "ordered_slice_coordinates": [
            float(value) if pd.notna(value) else None
            for value in ordered["SpatialSliceCoordinate"]
        ],
        "ordered_instance_numbers": [
            int(value) if pd.notna(value) else None for value in ordered["InstanceNumber"]
        ],
    }
    for column in SERIES_SUMMARY_COLUMNS:
        record[column] = _first_non_null(group[column]) if column in group else None
    return record


def build_series_manifest(
    metadata: pd.DataFrame,
    series_descriptors: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Create one ordered, auditable row per MRI series."""
    required = {
        "StudyInstanceUID",
        "SeriesInstanceUID",
        "SOPInstanceUID",
        "dicom_path",
        "SpatialSliceCoordinate",
        "InstanceNumber",
    }
    missing = sorted(required.difference(metadata.columns))
    if missing:
        raise ValueError(f"Metadata is missing required columns: {', '.join(missing)}")
    records = [
        _ordered_series_record(group)
        for _, group in metadata.groupby(
            ["StudyInstanceUID", "SeriesInstanceUID"],
            sort=True,
            dropna=False,
        )
    ]
    manifest = pd.DataFrame.from_records(records)
    if series_descriptors is not None:
        keys = ["StudyInstanceUID", "SeriesInstanceUID"]
        if series_descriptors.duplicated(keys).any():
            raise ValueError("Series descriptor table contains duplicate study/series rows")
        manifest = manifest.merge(
            series_descriptors,
            on=keys,
            how="left",
            validate="one_to_one",
        )
    return manifest


def _unique_json(values: Iterable[Any]) -> str:
    unique = sorted({str(value) for value in values if pd.notna(value)})
    return json.dumps(unique, ensure_ascii=False, separators=(",", ":"))


def build_extended_study_inventory(series_manifest: pd.DataFrame) -> pd.DataFrame:
    """Aggregate series properties and possible grouping identifiers per study."""
    required = {"StudyInstanceUID", "SeriesInstanceUID", "slice_count"}
    missing = sorted(required.difference(series_manifest.columns))
    if missing:
        raise ValueError(f"Series manifest is missing required columns: {', '.join(missing)}")
    records: list[dict[str, Any]] = []
    for study_id, group in series_manifest.groupby("StudyInstanceUID", sort=True, dropna=False):
        records.append(
            {
                "StudyInstanceUID": str(study_id),
                "series_count": int(group["SeriesInstanceUID"].nunique(dropna=True)),
                "slice_count": int(group["slice_count"].sum()),
                "geometry_orderable_series_fraction": float(
                    group["geometry_reliably_orderable"].mean()
                ),
                "PatientID": _first_non_null(group.get("PatientID", [])),
                "AccessionNumber": _first_non_null(group.get("AccessionNumber", [])),
                "manufacturers": _unique_json(group.get("Manufacturer", [])),
                "manufacturer_models": _unique_json(
                    group.get("ManufacturerModelName", [])
                ),
                "field_strengths": _unique_json(group.get("MagneticFieldStrength", [])),
                "anatomical_planes": _unique_json(group.get("Anatomical_Plane", [])),
                "fluid_sensitive_values": _unique_json(group.get("Fluid_Sensitive", [])),
                "fat_suppression_values": _unique_json(group.get("Fat_Suppression", [])),
            }
        )
    return pd.DataFrame.from_records(records)


def grouping_audit(study_inventory: pd.DataFrame) -> dict[str, Any]:
    """Assess whether de-identified patient or accession grouping survives."""
    result: dict[str, Any] = {}
    for column in ("PatientID", "AccessionNumber"):
        values = study_inventory[column] if column in study_inventory else pd.Series(dtype=object)
        non_null = values.dropna().astype(str)
        counts = non_null.value_counts()
        result[column] = {
            "non_null_studies": int(len(non_null)),
            "unique_values": int(non_null.nunique()),
            "values_shared_across_studies": int((counts > 1).sum()),
            "studies_in_shared_groups": int(counts[counts > 1].sum()),
            "maximum_studies_per_value": int(counts.max()) if not counts.empty else 0,
        }
    return result


def aggregate_metadata(
    metadata_parts_dir: str | Path,
    *,
    series_descriptors_path: str | Path,
    series_manifest_path: str | Path,
    study_inventory_path: str | Path,
    audit_path: str | Path,
) -> dict[str, Any]:
    """Aggregate all streamed parts and atomically write Stage 2 artifacts."""
    metadata = read_metadata_parts(metadata_parts_dir)
    descriptors = pd.read_csv(series_descriptors_path)
    series_manifest = build_series_manifest(metadata, descriptors)
    study_inventory = build_extended_study_inventory(series_manifest)
    audit = {
        **metadata_audit(metadata),
        "part_count": len(discover_metadata_parts(metadata_parts_dir)),
        "grouping": grouping_audit(study_inventory),
        "series_descriptor_missingness": {
            column: {
                "missing_count": int(series_manifest[column].isna().sum()),
                "missing_fraction": float(series_manifest[column].isna().mean()),
            }
            for column in ("Fluid_Sensitive", "Fat_Suppression", "Anatomical_Plane")
            if column in series_manifest
        },
        "order_method_distribution": {
            str(key): int(value)
            for key, value in series_manifest["order_method"].value_counts().items()
        },
    }

    outputs = (
        (series_manifest, Path(series_manifest_path).expanduser().resolve()),
        (study_inventory, Path(study_inventory_path).expanduser().resolve()),
    )
    for frame, output in outputs:
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        frame.to_parquet(temporary, index=False)
        temporary.replace(output)
    audit_output = Path(audit_path).expanduser().resolve()
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    temporary_audit = audit_output.with_suffix(audit_output.suffix + ".tmp")
    temporary_audit.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary_audit.replace(audit_output)
    return audit
