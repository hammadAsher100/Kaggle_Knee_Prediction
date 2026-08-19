"""Deterministic study-level validation and OOF assembly."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import torch

from src.training.metrics import multilabel_roc_auc


@torch.inference_mode()
def predict_loader(model, loader, *, device: torch.device, target_names: Sequence[str]):
    model.eval()
    identifiers: list[str] = []
    probabilities: list[np.ndarray] = []
    gold_targets: list[np.ndarray] = []
    for batch in loader:
        input_key = "features" if "features" in batch else "images"
        plane_ids = batch.get("plane_ids")
        if plane_ids is not None:
            plane_ids = plane_ids.to(device, non_blocking=True)
        output = model(
            batch[input_key].to(device, non_blocking=True),
            batch["slice_mask"].to(device, non_blocking=True),
            plane_ids,
        )
        probabilities.append(torch.sigmoid(output["logits"]).cpu().numpy())
        identifiers.extend([str(value) for value in batch["study_id"]])
        if "gold_targets" in batch:
            gold_targets.append(batch["gold_targets"].numpy())
    values = np.concatenate(probabilities) if probabilities else np.empty((0, len(target_names)))
    frame = pd.DataFrame(values, columns=[f"{target}__prediction" for target in target_names])
    frame.insert(0, "StudyInstanceUID", identifiers)
    metrics = None
    if gold_targets:
        truth = np.concatenate(gold_targets)
        metrics = multilabel_roc_auc(truth, values, target_names)
        for index, target in enumerate(target_names):
            frame[f"{target}__gold"] = truth[:, index]
    return frame, metrics
