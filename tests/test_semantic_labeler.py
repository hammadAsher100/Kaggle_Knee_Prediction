"""Tests for semantic score conversion and rule blending."""

from __future__ import annotations

import numpy as np
import pytest

from src.labeling.semantic_labeler import blend_probabilities, sigmoid_margin


def test_sigmoid_margin_is_bounded_and_monotonic() -> None:
    probabilities = sigmoid_margin(np.array([-1.0, 0.0, 1.0]))

    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))
    assert probabilities[0] < probabilities[1] < probabilities[2]
    assert probabilities[1] == pytest.approx(0.5)


def test_probability_blend_respects_weight_extremes() -> None:
    rules = np.array([[0.1, 0.9]])
    semantic = np.array([[0.7, 0.3]])

    assert np.array_equal(
        blend_probabilities(rules, semantic, rule_weight=1.0),
        rules,
    )
    assert np.array_equal(
        blend_probabilities(rules, semantic, rule_weight=0.0),
        semantic,
    )
    with pytest.raises(ValueError, match="between zero and one"):
        blend_probabilities(rules, semantic, rule_weight=1.5)
