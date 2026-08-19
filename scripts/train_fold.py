"""Train one leakage-safe attention head on cached DINOv2 features."""

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
from src.models.model_factory import StudyFeatureClassifier
from src.training.checkpoint import save_checkpoint
from src.training.optimizer import build_adamw
from src.training.scheduler import build_warmup_cosine_scheduler
from src.training.train_loop import train_one_epoch
from src.training.validation import predict_loader
from src.utils.seed import seed_everything

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
    parser.add_argument("--training-table", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gold-weight", type=float, default=4.0)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seed_everything(args.seed + args.fold)
    table = pd.read_parquet(args.training_table)
    for target in TARGETS:
        fold_target = f"{target}__train_fold_{args.fold}"
        if fold_target not in table:
            raise ValueError(f"Training table is missing nested target {fold_target}")
        table[f"{target}__train"] = table[fold_target]
    train_frame = table.loc[table["fold"] != args.fold].reset_index(drop=True)
    valid_frame = table.loc[table["fold"] == args.fold].reset_index(drop=True)
    if train_frame.empty or valid_frame.empty:
        raise ValueError(f"fold {args.fold} creates an empty train/validation split")
    train_dataset = FrozenFeatureDataset(train_frame, args.features, target_names=TARGETS)
    valid_dataset = FrozenFeatureDataset(valid_frame, args.features, target_names=TARGETS)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    feature_dim = int(train_dataset.features.shape[-1])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StudyFeatureClassifier(feature_dim, TARGETS).to(device)
    optimizer = build_adamw(
        model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    total_steps = args.epochs * max(len(train_loader), 1)
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        total_steps=total_steps,
        warmup_steps=max(total_steps // 10, 1),
    )
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    best_score = -float("inf")
    best_epoch = -1
    stale_epochs = 0
    history: list[dict[str, object]] = []
    for epoch in range(args.epochs):
        epoch_result = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device=device,
            scaler=scaler,
            scheduler=scheduler,
            gold_weight=args.gold_weight,
        )
        oof, metrics = predict_loader(model, valid_loader, device=device, target_names=TARGETS)
        score = metrics.macro_auc if metrics and metrics.macro_auc is not None else -float("inf")
        record = {
            "epoch": epoch,
            "train_loss": epoch_result["loss"],
            "gold_macro_auc": None if not np.isfinite(score) else score,
            "per_target_gold_auc": metrics.per_target if metrics else None,
        }
        history.append(record)
        print(json.dumps(record, allow_nan=False), flush=True)
        if best_epoch < 0 or score > best_score + 1e-6:
            best_score = score
            best_epoch = epoch
            stale_epochs = 0
            save_checkpoint(
                output / f"fold-{args.fold}.pt",
                {
                    "model_state": model.state_dict(),
                    "feature_dim": feature_dim,
                    "target_names": TARGETS,
                    "fold": args.fold,
                    "epoch": epoch,
                    "gold_macro_auc": None if not np.isfinite(score) else score,
                    "seed": args.seed,
                },
            )
            temporary_oof = output / f"fold-{args.fold}-oof.parquet.tmp"
            oof.to_parquet(temporary_oof, index=False)
            temporary_oof.replace(output / f"fold-{args.fold}-oof.parquet")
        else:
            stale_epochs += 1
        if stale_epochs >= args.patience:
            break
    metrics_path = output / f"fold-{args.fold}-history.json"
    metrics_path.write_text(
        json.dumps(
            {"fold": args.fold, "best_epoch": best_epoch, "history": history},
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
