"""Stage 5 metric test skeleton."""

import pytest

pytestmark = pytest.mark.skip(reason="Metric implementation is scheduled for Stage 5")


def test_macro_auc_all_negative_target_contract() -> None:
    """Will define the official behavior for targets absent from a validation fold."""
