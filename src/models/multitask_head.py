"""Target-schema-bound multitask prediction head."""

from __future__ import annotations

from collections.abc import Sequence

from torch import nn


class MultiTaskHead(nn.Module):
    """A compact shared projection followed by one logit per finding."""

    def __init__(
        self,
        feature_dim: int,
        target_names: Sequence[str],
        *,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if not target_names or len(set(target_names)) != len(target_names):
            raise ValueError("target_names must be non-empty and unique")
        self.target_names = tuple(target_names)
        self.layers = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, len(self.target_names)),
        )

    def forward(self, features):
        return self.layers(features)
