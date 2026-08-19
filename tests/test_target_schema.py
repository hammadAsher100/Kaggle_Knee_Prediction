"""Protect the official target and submission order across configurations."""

from __future__ import annotations

from pathlib import Path

from src.utils.config import load_config

TARGETS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]


def test_official_target_order_is_consistent() -> None:
    root = Path(__file__).resolve().parents[1]
    data = load_config(root / "configs" / "data.yaml")
    labeler = load_config(root / "configs" / "labeler.yaml")
    model_configs = [
        load_config(root / "configs" / name)
        for name in ("baseline.yaml", "dinov2.yaml", "convnext.yaml", "efficientnet.yaml")
    ]

    assert data["schema"]["target_columns"] == TARGETS
    assert data["submission"]["columns"] == ["StudyInstanceUID", *TARGETS]
    assert labeler["labeler"]["target_columns"] == TARGETS
    assert all(config["model"]["target_columns"] == TARGETS for config in model_configs)
