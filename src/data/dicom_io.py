"""Robust DICOM pixel decoding with explicit, auditable transforms."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class DicomDecodeError(RuntimeError):
    """Raised when an image cannot be decoded into a finite two-dimensional array."""


def decode_dicom(
    path: str | Path,
    *,
    apply_modality_lut: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Decode one DICOM slice and return float pixels plus useful provenance."""
    import pydicom

    resolved = Path(path).expanduser().resolve()
    try:
        dataset = pydicom.dcmread(resolved, force=True)
        pixels = np.asarray(dataset.pixel_array)
    except Exception as error:
        raise DicomDecodeError(
            f"Failed to decode DICOM {resolved}: {type(error).__name__}"
        ) from error
    if pixels.ndim == 3 and pixels.shape[0] == 1:
        pixels = pixels[0]
    if pixels.ndim != 2:
        raise DicomDecodeError(f"Expected a 2D MRI slice, got shape {pixels.shape}")
    values = pixels.astype(np.float32, copy=False)
    slope = float(getattr(dataset, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(dataset, "RescaleIntercept", 0.0) or 0.0)
    if apply_modality_lut:
        values = values * slope + intercept
    padding = getattr(dataset, "PixelPaddingValue", None)
    if padding is not None:
        padding_value = float(padding) * slope + intercept if apply_modality_lut else float(padding)
        values = values.copy()
        values[np.isclose(values, padding_value)] = np.nan
    if not np.isfinite(values).any():
        raise DicomDecodeError(f"DICOM has no finite image pixels: {resolved}")
    if str(getattr(dataset, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        finite = values[np.isfinite(values)]
        values = float(finite.max() + finite.min()) - values
    metadata = {
        "StudyInstanceUID": str(getattr(dataset, "StudyInstanceUID", "")),
        "SeriesInstanceUID": str(getattr(dataset, "SeriesInstanceUID", "")),
        "SOPInstanceUID": str(getattr(dataset, "SOPInstanceUID", "")),
        "TransferSyntaxUID": str(getattr(dataset.file_meta, "TransferSyntaxUID", "")),
        "PhotometricInterpretation": str(
            getattr(dataset, "PhotometricInterpretation", "")
        ),
        "RescaleSlope": slope,
        "RescaleIntercept": intercept,
    }
    return values, metadata
