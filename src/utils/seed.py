"""Deterministic seed configuration across available local libraries."""

from __future__ import annotations

import os
import random
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class SeedReport:
    """Records which random-number systems were configured."""

    seed: int
    deterministic: bool
    python: bool
    numpy: bool
    torch: bool
    cuda: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def set_global_seed(
    seed: int,
    *,
    deterministic: bool = True,
    include_numpy: bool = True,
    include_torch: bool = True,
) -> SeedReport:
    """Seed Python, NumPy, and PyTorch when requested and installed."""
    if seed < 0:
        raise ValueError("seed must be non-negative")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    numpy_seeded = False
    if include_numpy:
        try:
            import numpy as np
        except ImportError:
            pass
        else:
            np.random.seed(seed)
            numpy_seeded = True

    torch_seeded = False
    cuda_seeded = False
    if include_torch:
        try:
            import torch
        except ImportError:
            pass
        else:
            torch.manual_seed(seed)
            torch_seeded = True
            cuda_seeded = bool(torch.cuda.is_available())
            if cuda_seeded:
                torch.cuda.manual_seed_all(seed)
            if deterministic:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                if hasattr(torch, "use_deterministic_algorithms"):
                    torch.use_deterministic_algorithms(True, warn_only=True)

    return SeedReport(
        seed=seed,
        deterministic=deterministic,
        python=True,
        numpy=numpy_seeded,
        torch=torch_seeded,
        cuda=cuda_seeded,
    )

