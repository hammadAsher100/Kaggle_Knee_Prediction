"""Tests for Stage 2 metadata aggregation and slice-order manifests."""

from __future__ import annotations

import json

import pandas as pd

from src.data.metadata_aggregation import (
    build_extended_study_inventory,
    build_series_manifest,
    grouping_audit,
)


def _metadata_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "StudyInstanceUID": ["study-a"] * 3 + ["study-b"] * 2,
            "SeriesInstanceUID": ["series-a"] * 3 + ["series-b"] * 2,
            "SOPInstanceUID": ["sop-3", "sop-1", "sop-2", "sop-4", "sop-5"],
            "dicom_path": ["c.dcm", "a.dcm", "b.dcm", "d.dcm", "e.dcm"],
            "SpatialSliceCoordinate": [3.0, 1.0, 2.0, None, None],
            "InstanceNumber": [30, 10, 20, 2, 1],
            "PatientID": ["patient-x"] * 5,
            "AccessionNumber": ["acc-a"] * 3 + ["acc-b"] * 2,
            "Manufacturer": ["Vendor"] * 5,
            "ManufacturerModelName": ["Model"] * 5,
            "MagneticFieldStrength": [3.0] * 5,
        }
    )


def test_series_manifest_prefers_geometry_then_instance_number() -> None:
    descriptors = pd.DataFrame(
        {
            "StudyInstanceUID": ["study-a", "study-b"],
            "SeriesInstanceUID": ["series-a", "series-b"],
            "Fluid_Sensitive": [1, 0],
            "Fat_Suppression": [1, 0],
            "Anatomical_Plane": ["Sagittal", "Coronal"],
        }
    )

    manifest = build_series_manifest(_metadata_rows(), descriptors)
    first = manifest.set_index("SeriesInstanceUID").loc["series-a"]
    second = manifest.set_index("SeriesInstanceUID").loc["series-b"]

    assert first["order_method"] == "geometry"
    assert first["ordered_dicom_paths"] == ["a.dcm", "b.dcm", "c.dcm"]
    assert second["order_method"] == "instance_number"
    assert second["ordered_dicom_paths"] == ["e.dcm", "d.dcm"]
    assert first["Anatomical_Plane"] == "Sagittal"


def test_study_inventory_exposes_shared_patient_groups() -> None:
    manifest = build_series_manifest(_metadata_rows())
    inventory = build_extended_study_inventory(manifest)
    audit = grouping_audit(inventory)

    assert len(inventory) == 2
    assert audit["PatientID"]["unique_values"] == 1
    assert audit["PatientID"]["studies_in_shared_groups"] == 2
    assert json.loads(inventory.loc[0, "manufacturers"]) == ["Vendor"]
