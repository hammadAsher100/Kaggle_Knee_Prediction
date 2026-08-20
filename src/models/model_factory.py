"""Configuration-driven study-level model construction."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch
from torch import nn

from src.models.attention_pooling import MaskedAttentionPooling
from src.models.backbone import DinoV2Backbone
from src.models.multitask_head import MultiTaskHead


class StudyClassifier(nn.Module):
    """Encode 2.5D stacks, pool slices, and predict all findings jointly."""

    def __init__(
        self,
        backbone: nn.Module,
        feature_dim: int,
        target_names: Sequence[str],
        *,
        pooling: str = "attention",
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.pooling_name = pooling
        self.pool = MaskedAttentionPooling(feature_dim) if pooling == "attention" else None
        if pooling not in {"attention", "mean"}:
            raise ValueError(f"Unsupported pooling: {pooling}")
        self.head = MultiTaskHead(feature_dim, target_names, dropout=dropout)

    def forward(
        self,
        images: torch.Tensor,
        slice_mask: torch.Tensor | None = None,
        plane_ids: torch.Tensor | None = None,
    ):
        if images.ndim != 5:
            raise ValueError("images must have shape [batch, slices, channels, height, width]")
        batch, slices, channels, height, width = images.shape
        encoded = self.backbone(images.reshape(batch * slices, channels, height, width))
        encoded = encoded.reshape(batch, slices, -1)
        if self.pool is not None:
            pooled, attention = self.pool(encoded, slice_mask)
        else:
            if slice_mask is None:
                pooled = encoded.mean(dim=1)
            else:
                weights = slice_mask.to(encoded.dtype)
                pooled = (encoded * weights.unsqueeze(-1)).sum(dim=1) / weights.sum(
                    dim=1, keepdim=True
                ).clamp_min(1.0)
            attention = None
        return {"logits": self.head(pooled), "attention": attention}


def build_dinov2_study_model(
    model_path: str | Path,
    target_names: Sequence[str],
    *,
    trainable_blocks: int = 2,
    pooling: str = "attention",
    dropout: float = 0.2,
) -> StudyClassifier:
    backbone = DinoV2Backbone(model_path, trainable_blocks=trainable_blocks)
    return StudyClassifier(
        backbone,
        backbone.feature_dim,
        target_names,
        pooling=pooling,
        dropout=dropout,
    )


class StudyFeatureClassifier(nn.Module):
    """Trainable plane-aware attention head over frozen slice embeddings."""

    def __init__(
        self,
        feature_dim: int,
        target_names: Sequence[str],
        *,
        plane_embedding_dim: int = 16,
        attention_hidden_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.plane_embedding = nn.Embedding(4, plane_embedding_dim)
        combined_dim = feature_dim + plane_embedding_dim
        self.pool = MaskedAttentionPooling(combined_dim, hidden_dim=attention_hidden_dim)
        self.head = MultiTaskHead(combined_dim, target_names, dropout=dropout)

    def forward(
        self,
        features: torch.Tensor,
        slice_mask: torch.Tensor,
        plane_ids: torch.Tensor | None = None,
    ):
        if features.ndim != 3:
            raise ValueError("features must have shape [batch, slices, features]")
        if plane_ids is None:
            plane_ids = torch.zeros(features.shape[:2], dtype=torch.long, device=features.device)
        embedded = self.plane_embedding(plane_ids.clamp(0, 3))
        pooled, attention = self.pool(torch.cat([features, embedded], dim=-1), slice_mask)
        return {"logits": self.head(pooled), "attention": attention}


class TargetAttentionFeatureClassifier(nn.Module):
    """Pool different slices for each clinical target."""

    def __init__(
        self,
        feature_dim: int,
        target_names: Sequence[str],
        *,
        plane_embedding_dim: int = 16,
        attention_hidden_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.target_names = tuple(target_names)
        self.plane_embedding = nn.Embedding(4, plane_embedding_dim)
        combined_dim = feature_dim + plane_embedding_dim
        self.attention = nn.Sequential(
            nn.LayerNorm(combined_dim),
            nn.Linear(combined_dim, attention_hidden_dim),
            nn.Tanh(),
            nn.Linear(attention_hidden_dim, len(self.target_names)),
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier_weight = nn.Parameter(torch.empty(len(self.target_names), combined_dim))
        self.classifier_bias = nn.Parameter(torch.zeros(len(self.target_names)))
        nn.init.normal_(self.classifier_weight, mean=0.0, std=0.02)

    def forward(
        self,
        features: torch.Tensor,
        slice_mask: torch.Tensor,
        plane_ids: torch.Tensor | None = None,
    ):
        if features.ndim != 3:
            raise ValueError("features must have shape [batch, slices, features]")
        if slice_mask.shape != features.shape[:2]:
            raise ValueError("slice_mask shape does not match features")
        if (~slice_mask.bool().any(dim=1)).any():
            raise ValueError("each study must contain at least one valid slice")
        if plane_ids is None:
            plane_ids = torch.zeros(features.shape[:2], dtype=torch.long, device=features.device)
        combined = torch.cat(
            [features, self.plane_embedding(plane_ids.clamp(0, 3))],
            dim=-1,
        )
        scores = self.attention(combined)
        scores = scores.masked_fill(
            ~slice_mask.bool().unsqueeze(-1),
            torch.finfo(scores.dtype).min,
        )
        attention = torch.softmax(scores, dim=1)
        pooled = torch.einsum("bsd,bst->btd", combined, attention)
        logits = (self.dropout(pooled) * self.classifier_weight.unsqueeze(0)).sum(dim=-1)
        logits = logits + self.classifier_bias
        return {"logits": logits, "attention": attention}


def build_feature_classifier(
    architecture: str,
    feature_dim: int,
    target_names: Sequence[str],
    *,
    dropout: float = 0.2,
    attention_hidden_dim: int = 128,
) -> nn.Module:
    """Construct a serialized frozen-feature head by architecture name."""
    common = {
        "feature_dim": feature_dim,
        "target_names": target_names,
        "dropout": dropout,
        "attention_hidden_dim": attention_hidden_dim,
    }
    if architecture == "shared_attention":
        return StudyFeatureClassifier(**common)
    if architecture == "target_attention":
        return TargetAttentionFeatureClassifier(**common)
    raise ValueError(f"Unsupported feature architecture: {architecture}")
