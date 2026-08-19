"""Offline multilingual sentence-embedding label inference and blending."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

TARGET_HYPOTHESES: dict[str, tuple[str, str]] = {
    "ACL": ("The MRI shows an anterior cruciate ligament injury.", "The ACL is intact."),
    "MCL": ("The MRI shows a medial collateral ligament injury.", "The MCL is intact."),
    "Medial Meniscus": (
        "The MRI shows a medial meniscus tear.",
        "The medial meniscus is intact without tear.",
    ),
    "Lateral Meniscus": (
        "The MRI shows a lateral meniscus tear.",
        "The lateral meniscus is intact without tear.",
    ),
    "Medial OA": (
        "There is osteoarthritis in the medial tibiofemoral compartment.",
        "There is no medial compartment osteoarthritis.",
    ),
    "Lateral OA": (
        "There is osteoarthritis in the lateral tibiofemoral compartment.",
        "There is no lateral compartment osteoarthritis.",
    ),
    "PF OA": (
        "There is patellofemoral osteoarthritis.",
        "There is no patellofemoral osteoarthritis.",
    ),
    "Effusion": ("There is a knee joint effusion.", "There is no knee joint effusion."),
    "Synovitis": ("There is synovitis of the knee.", "There is no synovitis."),
    "Baker's": ("There is a Baker cyst.", "There is no Baker cyst."),
    "Contusion": ("There is a bone contusion or bone bruise.", "There is no bone contusion."),
    "Fracture": ("There is a fracture.", "There is no fracture."),
}


def sigmoid_margin(margin: np.ndarray, *, scale: float = 10.0) -> np.ndarray:
    """Map positive-minus-negative cosine margins to bounded probabilities."""
    clipped = np.clip(np.asarray(margin, dtype=np.float64) * scale, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def blend_probabilities(
    rule_probabilities: np.ndarray,
    semantic_probabilities: np.ndarray,
    *,
    rule_weight: float,
) -> np.ndarray:
    if not 0.0 <= rule_weight <= 1.0:
        raise ValueError("rule_weight must be between zero and one")
    rules = np.asarray(rule_probabilities, dtype=np.float64)
    semantic = np.asarray(semantic_probabilities, dtype=np.float64)
    if rules.shape != semantic.shape:
        raise ValueError("Rule and semantic probability arrays must have the same shape")
    return rule_weight * rules + (1.0 - rule_weight) * semantic


def encode_semantic_probabilities(
    model: Any,
    report_sentences: Sequence[Sequence[str]],
    *,
    target_columns: Sequence[str],
    batch_size: int = 256,
    margin_scale: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Encode report sentences and return semantic probabilities and prompt scores."""
    missing = [target for target in target_columns if target not in TARGET_HYPOTHESES]
    if missing:
        raise KeyError("Missing semantic hypotheses: " + ", ".join(missing))

    flattened: list[str] = []
    offsets: list[tuple[int, int]] = []
    for sentences in report_sentences:
        start = len(flattened)
        flattened.extend(sentences or ("No report text.",))
        offsets.append((start, len(flattened)))
    embeddings = np.asarray(
        model.encode(
            flattened,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        ),
        dtype=np.float32,
    )
    prompts = [
        hypothesis
        for target in target_columns
        for hypothesis in TARGET_HYPOTHESES[target]
    ]
    prompt_embeddings = np.asarray(
        model.encode(
            prompts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ),
        dtype=np.float32,
    ).reshape(len(target_columns), 2, -1)

    positive_scores = np.empty((len(report_sentences), len(target_columns)), dtype=np.float32)
    negative_scores = np.empty_like(positive_scores)
    for report_index, (start, end) in enumerate(offsets):
        report_embeddings = embeddings[start:end]
        for target_index in range(len(target_columns)):
            positive_scores[report_index, target_index] = float(
                np.max(report_embeddings @ prompt_embeddings[target_index, 0])
            )
            negative_scores[report_index, target_index] = float(
                np.max(report_embeddings @ prompt_embeddings[target_index, 1])
            )
    probabilities = sigmoid_margin(
        positive_scores - negative_scores,
        scale=margin_scale,
    )
    return probabilities, positive_scores, negative_scores


def stable_model_identifier(
    model_path: str,
    model_metadata: Mapping[str, Any] | None = None,
) -> str:
    """Create a loggable model identifier without embedding machine-specific paths."""
    if model_metadata and model_metadata.get("id"):
        return str(model_metadata["id"])
    normalized = model_path.replace("\\", "/").rstrip("/")
    return normalized.rsplit("/", maxsplit=1)[-1] or "unknown-model"


def finite_probability(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError("Semantic label probability must be finite")
    return min(1.0, max(0.0, float(value)))
