"""Tests for the frozen-feature study head and soft-label loss."""

import pytest

torch = pytest.importorskip("torch")

from src.models.model_factory import (  # noqa: E402
    StudyFeatureClassifier,
    TargetAttentionFeatureClassifier,
    build_feature_classifier,
)
from src.training.losses import weighted_soft_bce  # noqa: E402


def test_feature_model_respects_mask_and_target_width() -> None:
    model = StudyFeatureClassifier(8, ["ACL", "MCL"], plane_embedding_dim=4)
    features = torch.randn(3, 5, 8)
    mask = torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 1, 1], [1, 0, 0, 0, 0]]).bool()
    planes = torch.tensor([[1, 1, 0, 0, 0], [1, 2, 3, 1, 2], [3, 0, 0, 0, 0]])
    output = model(features, mask, planes)
    assert output["logits"].shape == (3, 2)
    assert output["attention"].shape == (3, 5)
    assert torch.allclose(output["attention"].sum(dim=1), torch.ones(3))
    assert output["attention"][0, 2:].eq(0).all()


def test_weighted_soft_bce_accepts_soft_targets() -> None:
    logits = torch.zeros((2, 2), requires_grad=True)
    targets = torch.tensor([[0.0, 0.25], [1.0, 0.75]])
    gold = torch.tensor([[1, 0], [1, 0]]).bool()
    loss = weighted_soft_bce(logits, targets, gold, gold_weight=4.0)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None


def test_target_attention_has_separate_normalized_target_weights() -> None:
    model = TargetAttentionFeatureClassifier(8, ["ACL", "MCL"], plane_embedding_dim=4)
    features = torch.randn(3, 5, 8)
    mask = torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 1, 1], [1, 0, 0, 0, 0]]).bool()
    planes = torch.tensor([[1, 1, 0, 0, 0], [1, 2, 3, 1, 2], [3, 0, 0, 0, 0]])

    output = model(features, mask, planes)

    assert output["logits"].shape == (3, 2)
    assert output["attention"].shape == (3, 5, 2)
    assert torch.allclose(output["attention"].sum(dim=1), torch.ones(3, 2))
    assert output["attention"][0, 2:].eq(0).all()


def test_feature_classifier_factory_rejects_unknown_architecture() -> None:
    with pytest.raises(ValueError, match="Unsupported feature architecture"):
        build_feature_classifier("unknown", 8, ["ACL"])
