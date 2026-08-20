import numpy as np
import pytest

from src.training.multimodal_fusion import nested_simplex_fusion, simplex_weights


def test_simplex_weights_cover_valid_combinations() -> None:
    weights = simplex_weights(3, 0.5)

    assert weights.shape == (6, 3)
    assert np.allclose(weights.sum(axis=1), 1)
    assert np.all(weights >= 0)


def test_simplex_weights_reject_invalid_step() -> None:
    with pytest.raises(ValueError, match="divide 1 exactly"):
        simplex_weights(3, 0.3)


def test_nested_fusion_selects_weights_without_held_out_fold() -> None:
    folds = np.repeat(np.arange(3), 4)
    labels = np.tile([0.0, 1.0, 0.0, 1.0], 3)[:, None]
    good = np.tile([0.1, 0.9, 0.2, 0.8], 3)[:, None]
    bad = 1 - good
    predictions = np.stack([good, bad], axis=2)

    result = nested_simplex_fusion(labels, predictions, folds, step=0.5)

    assert result.macro_auc == pytest.approx(1.0)
    assert all(np.allclose(weight, [1.0, 0.0]) for weight in result.weights.values())
    assert result.predictions.shape == labels.shape


def test_nested_fusion_supports_target_specific_weights() -> None:
    folds = np.repeat(np.arange(3), 4)
    labels = np.column_stack(
        [np.tile([0.0, 1.0, 0.0, 1.0], 3), np.tile([1.0, 0.0, 1.0, 0.0], 3)]
    )
    first = np.column_stack([labels[:, 0], 1 - labels[:, 1]])
    second = np.column_stack([1 - labels[:, 0], labels[:, 1]])
    predictions = np.stack([first, second], axis=2)

    result = nested_simplex_fusion(labels, predictions, folds, step=0.5, per_target=True)

    assert result.macro_auc == pytest.approx(1.0)
    for weights in result.weights.values():
        assert np.allclose(weights, [[1.0, 0.0], [0.0, 1.0]])
