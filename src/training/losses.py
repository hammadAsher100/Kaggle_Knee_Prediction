"""Losses for soft weak labels with stronger gold supervision."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def weighted_soft_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gold_mask: torch.Tensor | None = None,
    *,
    gold_weight: float = 4.0,
    target_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Binary cross entropy supporting soft labels and per-cell gold emphasis."""
    if logits.shape != targets.shape:
        raise ValueError("logits and targets must have identical shapes")
    if not torch.isfinite(targets).all() or ((targets < 0) | (targets > 1)).any():
        raise ValueError("targets must be finite probabilities")
    losses = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    weights = torch.ones_like(losses)
    if gold_mask is not None:
        if gold_mask.shape != losses.shape:
            raise ValueError("gold_mask must match logits")
        weights = torch.where(gold_mask.bool(), gold_weight, 1.0)
    if target_weights is not None:
        if target_weights.numel() != losses.shape[-1]:
            raise ValueError("target_weights width does not match targets")
        weights = weights * target_weights.reshape(1, -1)
    return (losses * weights).sum() / weights.sum().clamp_min(1.0)
