"""Stage 4 slice sampler test skeleton."""

import pytest

pytestmark = pytest.mark.skip(reason="Slice sampling implementation is scheduled for Stage 4")


def test_slice_sampler_boundary_contract() -> None:
    """Will cover short-series boundaries and deterministic validation indices."""

