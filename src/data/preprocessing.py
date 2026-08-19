"""MRI-specific intensity preprocessing."""

from __future__ import annotations

import numpy as np


def robust_window(image: np.ndarray, lower: float = 0.5, upper: float = 99.5) -> np.ndarray:
    """Clip finite pixels to robust percentiles while preserving shape."""
    values = np.asarray(image, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values, dtype=np.float32)
    low, high = np.percentile(finite, [lower, upper])
    if high <= low:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip(np.nan_to_num(values, nan=low, posinf=high, neginf=low), low, high)


def normalize_mri(image: np.ndarray) -> np.ndarray:
    """Robustly scale an MRI slice to [0, 1]."""
    clipped = robust_window(image)
    low = float(clipped.min())
    high = float(clipped.max())
    if high <= low:
        return np.zeros_like(clipped, dtype=np.float32)
    return ((clipped - low) / (high - low)).astype(np.float32)


def center_crop_or_pad(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Center crop and zero-pad the last two dimensions to an exact size."""
    values = np.asarray(image)
    target_h, target_w = size
    if target_h < 1 or target_w < 1 or values.ndim < 2:
        raise ValueError("image and target size must be at least two-dimensional")
    source_h, source_w = values.shape[-2:]
    crop_h, crop_w = min(source_h, target_h), min(source_w, target_w)
    source_y = max((source_h - crop_h) // 2, 0)
    source_x = max((source_w - crop_w) // 2, 0)
    cropped = values[..., source_y : source_y + crop_h, source_x : source_x + crop_w]
    result = np.zeros((*values.shape[:-2], target_h, target_w), dtype=values.dtype)
    target_y = (target_h - crop_h) // 2
    target_x = (target_w - crop_w) // 2
    result[..., target_y : target_y + crop_h, target_x : target_x + crop_w] = cropped
    return result
