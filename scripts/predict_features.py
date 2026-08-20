"""Ensemble fold heads over cached study features."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import FrozenFeatureDataset
from src.inference.calibration import apply_multilabel_calibration
from src.inference.ensemble import weighted_probability_average
from src.models.model_factory import build_feature_classifier
from src.training.checkpoint import load_checkpoint


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-table", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--calibration")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    studies = (
        pd.read_csv(args.study_table)
        if Path(args.study_table).suffix.lower() == ".csv"
        else pd.read_parquet(args.study_table)
    )
    dataset = FrozenFeatureDataset(studies, args.features)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    checkpoints = sorted(Path(args.checkpoint_dir).glob("fold-*.pt"))
    if len(checkpoints) != 5:
        raise ValueError(f"Expected five fold checkpoints, found {len(checkpoints)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fold_outputs: list[np.ndarray] = []
    target_names: list[str] | None = None
    for path in checkpoints:
        payload = load_checkpoint(path)
        names = list(payload["target_names"])
        if target_names is not None and names != target_names:
            raise ValueError("Checkpoint target orders differ")
        target_names = names
        model = build_feature_classifier(
            str(payload.get("architecture", "shared_attention")),
            int(payload["feature_dim"]),
            names,
            dropout=float(payload.get("dropout", 0.2)),
            attention_hidden_dim=int(payload.get("attention_hidden_dim", 128)),
        ).to(device)
        model.load_state_dict(payload["model_state"], strict=True)
        model.eval()
        fold_predictions: list[np.ndarray] = []
        with torch.inference_mode():
            for batch in loader:
                output = model(
                    batch["features"].to(device, non_blocking=True),
                    batch["slice_mask"].to(device, non_blocking=True),
                    batch["plane_ids"].to(device, non_blocking=True),
                )
                fold_predictions.append(torch.sigmoid(output["logits"]).cpu().numpy())
        values = np.concatenate(fold_predictions)
        fold_outputs.append(values)
    assert target_names is not None
    ensemble = weighted_probability_average(fold_outputs)
    if args.calibration:
        parameters = json.loads(Path(args.calibration).read_text(encoding="utf-8"))
        ensemble = apply_multilabel_calibration(ensemble, target_names, parameters)
    output_frame = pd.DataFrame(
        ensemble,
        columns=[f"{target}__prediction" for target in target_names],
    )
    output_frame.insert(0, "StudyInstanceUID", studies["StudyInstanceUID"].astype(str))
    output = Path(args.output)
    temporary = output.with_suffix(output.suffix + ".tmp")
    output_frame.to_parquet(temporary, index=False)
    temporary.replace(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
