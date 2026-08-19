"""Tests for fast geometry-ordered inference manifests."""

from pathlib import Path

import pandas as pd
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

from src.data.inference_manifest import build_inference_series_manifest


def _write_header(path: Path, study: str, series: str, instance: int, z: float) -> None:
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = MRImageStorage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dataset = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    dataset.StudyInstanceUID = study
    dataset.SeriesInstanceUID = series
    dataset.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    dataset.InstanceNumber = instance
    dataset.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    dataset.ImagePositionPatient = [0, 0, z]
    dataset.save_as(path, enforce_file_format=True)


def test_inference_manifest_orders_by_geometry(tmp_path: Path) -> None:
    study, series = generate_uid(), generate_uid()
    directory = tmp_path / study / series
    directory.mkdir(parents=True)
    _write_header(directory / "a.dcm", study, series, 1, 3.0)
    _write_header(directory / "b.dcm", study, series, 2, 1.0)
    descriptors = pd.DataFrame(
        {
            "StudyInstanceUID": [study],
            "SeriesInstanceUID": [series],
            "Fluid_Sensitive": [1],
            "Fat_Suppression": [0],
            "Anatomical_Plane": ["Sagittal"],
        }
    )
    manifest, audit = build_inference_series_manifest(descriptors, tmp_path)
    assert manifest.loc[0, "ordered_dicom_paths"] == [
        f"{study}/{series}/b.dcm",
        f"{study}/{series}/a.dcm",
    ]
    assert manifest.loc[0, "order_method"] == "geometry"
    assert audit["header_failure_count"] == 0
