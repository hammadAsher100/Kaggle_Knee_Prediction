"""Learning-rate scheduler construction."""

from __future__ import annotations

import math

from torch.optim.lr_scheduler import LambdaLR


def build_warmup_cosine_scheduler(optimizer, *, total_steps: int, warmup_steps: int):
    if total_steps < 1 or not 0 <= warmup_steps < total_steps:
        raise ValueError("scheduler steps are invalid")

    def scale(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return LambdaLR(optimizer, scale)
