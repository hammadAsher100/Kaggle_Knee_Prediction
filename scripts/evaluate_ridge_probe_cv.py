"""Evaluate a nested-CV linear probe on frozen study-level DINOv2 features."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.multimodal_fusion import score_targets

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--training-table", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-oof", type=Path, required=True)
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.1, 1.0, 10.0, 100.0])
    return parser.parse_args()


def masked_mean(features: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Pool valid slice features into one vector per study."""
    weights = mask.astype(np.float32)[..., None]
    denominator = np.maximum(weights.sum(axis=1), 1.0)
    return (features.astype(np.float32) * weights).sum(axis=1) / denominator


def fit_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    alpha: float,
) -> np.ndarray:
    model = make_pipeline(
        StandardScaler(),
        Ridge(alpha=alpha, solver="lsqr"),
    )
    model.fit(x_train, y_train)
    return model.predict(x_valid)


def main() -> None:
    args = parse_args()
    archive = np.load(args.features, allow_pickle=False)
    feature_frame = pd.DataFrame(
        {
            "StudyInstanceUID": archive["study_ids"].astype(str),
            "feature_index": np.arange(len(archive["study_ids"])),
        }
    )
    x_all = masked_mean(archive["features"], archive["slice_mask"])
    table = pd.read_parquet(args.training_table).merge(
        feature_frame,
        on="StudyInstanceUID",
        how="inner",
        validate="one_to_one",
    )
    if len(table) != len(feature_frame):
        raise ValueError("training table and features do not contain identical study IDs")
    x = x_all[table["feature_index"].to_numpy(int)]
    gold = table[[f"{target}__gold" for target in TARGETS]].to_numpy(float)
    folds = table["fold"].to_numpy(int)
    oof = np.full(gold.shape, np.nan, dtype=np.float64)
    selections: dict[str, dict[str, object]] = {}

    for outer_fold in np.unique(folds):
        development = folds != outer_fold
        outer_valid = folds == outer_fold
        train_targets = table[
            [f"{target}__train_fold_{outer_fold}" for target in TARGETS]
        ].to_numpy(float)
        candidate_scores: dict[str, float] = {}
        for alpha in args.alphas:
            inner_oof = np.full(gold.shape, np.nan, dtype=np.float64)
            for inner_fold in np.unique(folds[development]):
                inner_train = development & (folds != inner_fold)
                inner_valid = development & (folds == inner_fold)
                inner_oof[inner_valid] = fit_predict(
                    x[inner_train], train_targets[inner_train], x[inner_valid], alpha
                )
            candidate_scores[str(alpha)] = score_targets(
                gold[development], inner_oof[development]
            )[0]
        selected_alpha = max(args.alphas, key=lambda value: candidate_scores[str(value)])
        oof[outer_valid] = fit_predict(
            x[development], train_targets[development], x[outer_valid], selected_alpha
        )
        selections[str(int(outer_fold))] = {
            "selected_alpha": selected_alpha,
            "inner_macro_auc": candidate_scores[str(selected_alpha)],
            "candidate_scores": candidate_scores,
        }

    macro_auc, per_target_auc = score_targets(gold, oof)
    payload = {
        "evaluation": "nested_oof_ridge_probe",
        "macro_auc": macro_auc,
        "per_target": dict(zip(TARGETS, per_target_auc.tolist(), strict=True)),
        "outer_fold_selections": selections,
        "gold_rows": int(np.isfinite(gold).all(axis=1).sum()),
        "pooling": "masked_mean",
        "limitations": [
            "Only 58 studies have gold labels, so the estimate has high variance.",
            "Hyperparameters are selected only on the development portion of each outer fold.",
            "This is a local cross-validation result, not a Kaggle leaderboard score.",
        ],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    output = table[["StudyInstanceUID", "fold"]].copy()
    for index, target in enumerate(TARGETS):
        output[f"{target}__prediction"] = oof[:, index]
        output[f"{target}__gold"] = gold[:, index]
    args.output_oof.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(args.output_oof, index=False)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
