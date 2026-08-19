"""Tests for official-style multilabel ROC AUC."""

import numpy as np

from src.training.metrics import multilabel_roc_auc


def test_macro_auc_all_negative_target_contract() -> None:
    truth = np.asarray([[0, 0], [1, 0], [np.nan, 0]], dtype=float)
    probabilities = np.asarray([[0.1, 0.3], [0.9, 0.2], [0.5, 0.1]])
    result = multilabel_roc_auc(truth, probabilities, ["variable", "all_negative"])
    assert result.per_target == {"variable": 1.0, "all_negative": None}
    assert result.macro_auc == 1.0
    assert result.valid_target_count == 1
