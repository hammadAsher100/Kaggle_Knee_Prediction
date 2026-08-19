"""Tests for mandatory pre-cache storage estimation."""

from __future__ import annotations

import pytest

from src.data.storage import compare_representations, estimate_cache_size


def test_cache_estimate_matches_dimension_product() -> None:
    estimate = estimate_cache_size(
        representation="fp16",
        studies=10,
        series_per_study=2,
        slices_per_series=16,
        height=224,
        width=224,
    )

    assert estimate.estimated_bytes == 10 * 2 * 16 * 224 * 224 * 2
    assert estimate.bytes_per_pixel == 2
    assert estimate.estimated_gib == pytest.approx(estimate.estimated_bytes / 1024**3)


def test_representation_comparison_is_explicit() -> None:
    estimates = compare_representations(
        studies=100,
        series_per_study=1,
        slices_per_series=8,
        height=128,
        width=128,
    )

    sizes = {estimate.representation: estimate.estimated_bytes for estimate in estimates}
    assert sizes["fp32"] == 2 * sizes["fp16"]
    assert sizes["fp16"] == sizes["uint16"]
    assert sizes["uint16"] == 2 * sizes["uint8"]


@pytest.mark.parametrize(
    "arguments",
    [
        {"representation": "int8"},
        {"studies": 0},
        {"compression_ratio": 0},
        {"compression_ratio": 1.1},
    ],
)
def test_cache_estimator_rejects_invalid_plans(arguments: dict[str, object]) -> None:
    valid: dict[str, object] = {
        "representation": "fp16",
        "studies": 10,
        "series_per_study": 1,
        "slices_per_series": 8,
        "height": 128,
        "width": 128,
        "compression_ratio": 1.0,
    }
    valid.update(arguments)

    with pytest.raises(ValueError):
        estimate_cache_size(**valid)
