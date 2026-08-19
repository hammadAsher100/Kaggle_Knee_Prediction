"""Tests for auditable series selection."""

from src.data.series_selector import categorize_series, select_series


def test_series_categorization_contract() -> None:
    row = {
        "SeriesInstanceUID": "sag",
        "Anatomical_Plane": "Sagittal",
        "Fluid_Sensitive": 1,
        "Fat_Suppression": 1,
        "slice_count": 30,
        "geometry_reliably_orderable": True,
    }
    assert categorize_series(row) == "sagittal_fluid_fs"
    selected = select_series(
        [
            row,
            {**row, "SeriesInstanceUID": "cor", "Anatomical_Plane": "Coronal"},
            {**row, "SeriesInstanceUID": "ax", "Anatomical_Plane": "Axial"},
            {**row, "SeriesInstanceUID": "extra", "slice_count": 1},
        ],
        max_series=3,
    )
    assert {item.series_uid for item in selected} == {"sag", "cor", "ax"}
