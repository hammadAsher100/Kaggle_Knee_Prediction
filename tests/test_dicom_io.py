"""Tests for pixel-free DICOM header extraction and geometry inventory."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

from src.data.metadata import (
    build_study_inventory,
    extract_metadata,
    metadata_audit,
    read_dicom_metadata,
    spatial_slice_coordinate,
)


def _write_dicom(
    path: Path,
    *,
    study_uid: str,
    series_uid: str,
    sop_uid: str,
    instance: int,
    z_position: float,
) -> None:
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = MRImageStorage
    file_meta.MediaStorageSOPInstanceUID = sop_uid
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dataset = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = MRImageStorage
    dataset.SOPInstanceUID = sop_uid
    dataset.StudyInstanceUID = study_uid
    dataset.SeriesInstanceUID = series_uid
    dataset.InstanceNumber = instance
    dataset.SeriesDescription = "SAG PD FS"
    dataset.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    dataset.ImagePositionPatient = [0, 0, z_position]
    dataset.Rows = 64
    dataset.Columns = 80
    dataset.Manufacturer = "Synthetic Vendor"
    dataset.save_as(path, enforce_file_format=True)


def test_spatial_slice_coordinate_uses_geometry() -> None:
    assert spatial_slice_coordinate([1, 0, 0, 0, 1, 0], [0, 0, 7.5]) == 7.5
    assert spatial_slice_coordinate(None, [0, 0, 1]) is None
    assert spatial_slice_coordinate([1, 0], [0, 0, 1]) is None


def test_extract_metadata_records_failures_and_builds_inventory(tmp_path: Path) -> None:
    root = tmp_path / "dicoms"
    series_dir = root / "study-a" / "series-a"
    series_dir.mkdir(parents=True)
    study_uid = generate_uid()
    series_uid = generate_uid()
    for instance, z_position in enumerate((4.0, 2.0, 6.0), start=1):
        _write_dicom(
            series_dir / f"arbitrary-{instance}.bin",
            study_uid=study_uid,
            series_uid=series_uid,
            sop_uid=generate_uid(),
            instance=instance,
            z_position=z_position,
        )
    (root / "not-a-dicom.txt").write_text("broken", encoding="utf-8")
    output = tmp_path / "metadata.parquet"

    result = extract_metadata(root, output)
    metadata = pd.read_parquet(output)
    inventory = build_study_inventory(metadata)

    assert result.discovered_files == 4
    assert result.readable_dicoms == 3
    assert result.failed_files == 1
    assert sorted(metadata["SpatialSliceCoordinate"].tolist()) == [2.0, 4.0, 6.0]
    assert inventory.loc[0, "series_count"] == 1
    assert inventory.loc[0, "slice_count"] == 3
    assert inventory.loc[0, "geometry_slice_fraction"] == 1.0
    assert Path(result.failure_path or "").read_text(encoding="utf-8").count("\n") == 1
    audit = metadata_audit(metadata)
    assert audit["dicom_count"] == 3
    assert audit["study_count"] == 1
    assert audit["series_count"] == 1
    assert audit["geometry"]["series_reliably_orderable"] == 1
    assert audit["categorical_distributions"]["Manufacturer"] == {
        "Synthetic Vendor": 3
    }


def test_read_dicom_metadata_preserves_relative_path(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    path = root / "slice.no_extension"
    _write_dicom(
        path,
        study_uid=generate_uid(),
        series_uid=generate_uid(),
        sop_uid=generate_uid(),
        instance=1,
        z_position=1.0,
    )

    record = read_dicom_metadata(path, root=root)

    assert record["dicom_path"] == "slice.no_extension"
    assert record["Rows"] == 64
    assert record["Columns"] == 80
