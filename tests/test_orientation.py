"""Stage 4 orientation test skeleton."""

import pytest

pytestmark = pytest.mark.skip(reason="Orientation logic requires the Stage 2 geometry audit")


def test_orientation_normalization_contract() -> None:
    """Will cover geometry-derived orientation without filename assumptions."""

