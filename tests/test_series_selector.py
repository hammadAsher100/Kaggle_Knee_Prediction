"""Stage 4 series selector test skeleton."""

import pytest

pytestmark = pytest.mark.skip(reason="Series rules require observed DICOM metadata from Stage 2")


def test_series_categorization_contract() -> None:
    """Will cover plane, sequence, fat suppression, other, and unknown classes."""

