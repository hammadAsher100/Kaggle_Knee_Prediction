"""Fast test-series ordering from DICOM geometry and competition descriptors."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.data.metadata import spatial_slice_coordinate


def build_inference_series_manifest(
    descriptors: pd.DataFrame,
    dicom_root: str | Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read only ordering headers and return one row per descriptor series."""
    import pydicom

    root = Path(dicom_root).expanduser().resolve()
    records: list[dict[str, Any]] = []
    failure_count = 0
    for descriptor in descriptors.to_dict("records"):
        study_id = str(descriptor["StudyInstanceUID"])
        series_id = str(descriptor["SeriesInstanceUID"])
        directory = (root / study_id / series_id).resolve()
        try:
            directory.relative_to(root)
        except ValueError as error:
            raise ValueError("Series path escapes DICOM root") from error
        files = sorted(path for path in directory.iterdir() if path.is_file())
        slices: list[dict[str, Any]] = []
        for path in files:
            coordinate = None
            instance = None
            try:
                dataset = pydicom.dcmread(
                    path,
                    stop_before_pixels=True,
                    force=True,
                    specific_tags=[
                        "ImageOrientationPatient",
                        "ImagePositionPatient",
                        "InstanceNumber",
                    ],
                )
                coordinate = spatial_slice_coordinate(
                    getattr(dataset, "ImageOrientationPatient", None),
                    getattr(dataset, "ImagePositionPatient", None),
                )
                raw_instance = getattr(dataset, "InstanceNumber", None)
                instance = int(raw_instance) if raw_instance is not None else None
            except Exception:
                failure_count += 1
            slices.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "coordinate": coordinate,
                    "instance": instance,
                }
            )
        geometry = bool(slices) and all(item["coordinate"] is not None for item in slices)
        unique_geometry = geometry and len({item["coordinate"] for item in slices}) == len(slices)
        instances = bool(slices) and all(item["instance"] is not None for item in slices)
        unique_instances = instances and len({item["instance"] for item in slices}) == len(slices)
        if unique_geometry:
            slices.sort(key=lambda item: (item["coordinate"], item["path"]))
            order_method = "geometry"
        elif unique_instances:
            slices.sort(key=lambda item: (item["instance"], item["path"]))
            order_method = "instance_number"
        else:
            slices.sort(key=lambda item: item["path"])
            order_method = "path_fallback"
        records.append(
            {
                **descriptor,
                "slice_count": len(slices),
                "geometry_reliably_orderable": unique_geometry,
                "instance_reliably_orderable": unique_instances,
                "order_method": order_method,
                "ordered_dicom_paths": [item["path"] for item in slices],
            }
        )
    manifest = pd.DataFrame.from_records(records)
    audit = {
        "study_count": int(manifest["StudyInstanceUID"].nunique()),
        "series_count": len(manifest),
        "slice_count": int(manifest["slice_count"].sum()),
        "header_failure_count": failure_count,
        "empty_series_count": int(manifest["slice_count"].eq(0).sum()),
        "order_method_distribution": {
            str(key): int(value) for key, value in manifest["order_method"].value_counts().items()
        },
    }
    return manifest, audit
