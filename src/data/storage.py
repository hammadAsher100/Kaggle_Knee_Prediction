"""Pre-materialization storage estimates for proposed image caches."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

BYTES_PER_PIXEL = {
    "fp32": 4,
    "fp16": 2,
    "uint16": 2,
    "uint8": 1,
}


@dataclass(frozen=True)
class CacheEstimate:
    """Estimated uncompressed or ratio-adjusted cache footprint."""

    representation: str
    studies: int
    series_per_study: float
    slices_per_series: float
    height: int
    width: int
    bytes_per_pixel: int
    compression_ratio: float
    estimated_bytes: int
    estimated_gib: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def estimate_cache_size(
    *,
    representation: str,
    studies: int,
    series_per_study: float,
    slices_per_series: float,
    height: int,
    width: int,
    compression_ratio: float = 1.0,
) -> CacheEstimate:
    """Estimate cache size before any expensive image conversion begins."""
    normalized = representation.lower()
    if normalized not in BYTES_PER_PIXEL:
        raise ValueError(f"Unsupported representation: {representation}")
    numeric_values = (studies, series_per_study, slices_per_series, height, width)
    if any(value <= 0 for value in numeric_values):
        raise ValueError("All cache dimensions must be positive")
    if compression_ratio <= 0 or compression_ratio > 1:
        raise ValueError("compression_ratio must be in (0, 1]")
    bytes_per_pixel = BYTES_PER_PIXEL[normalized]
    raw = (
        studies
        * series_per_study
        * slices_per_series
        * height
        * width
        * bytes_per_pixel
    )
    estimated_bytes = int(round(raw * compression_ratio))
    return CacheEstimate(
        representation=normalized,
        studies=studies,
        series_per_study=series_per_study,
        slices_per_series=slices_per_series,
        height=height,
        width=width,
        bytes_per_pixel=bytes_per_pixel,
        compression_ratio=compression_ratio,
        estimated_bytes=estimated_bytes,
        estimated_gib=estimated_bytes / 1024**3,
    )


def compare_representations(**dimensions: Any) -> list[CacheEstimate]:
    """Return an explicit FP32/FP16/UINT16/UINT8 size comparison."""
    return [
        estimate_cache_size(representation=representation, **dimensions)
        for representation in BYTES_PER_PIXEL
    ]
