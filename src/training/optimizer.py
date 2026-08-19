"""Optimizer construction."""

from __future__ import annotations

import torch


def build_adamw(model, *, learning_rate: float, weight_decay: float):
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("model has no trainable parameters")
    return torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
