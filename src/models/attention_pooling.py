"""Masked slice attention pooling."""

from __future__ import annotations

import torch
from torch import nn


class MaskedAttentionPooling(nn.Module):
    """Learn a scalar importance weight per valid slice."""

    def __init__(self, feature_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.scorer = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        features: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if features.ndim != 3:
            raise ValueError("features must have shape [batch, slices, features]")
        scores = self.scorer(features).squeeze(-1)
        if mask is not None:
            if mask.shape != scores.shape:
                raise ValueError("mask shape does not match slice dimensions")
            valid = mask.bool()
            if (~valid.any(dim=1)).any():
                raise ValueError("each study must contain at least one valid slice")
            scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1)
        return torch.sum(features * weights.unsqueeze(-1), dim=1), weights
