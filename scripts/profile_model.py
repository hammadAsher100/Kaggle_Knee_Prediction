"""Profile a trained feature head's size, latency, and runtime environment."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.model_factory import StudyFeatureClassifier
from src.training.checkpoint import load_checkpoint
from src.utils.hardware import runtime_metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--slices", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--attention-hidden-dim", type=int)
    args = parser.parse_args()
    payload = load_checkpoint(args.checkpoint)
    model = StudyFeatureClassifier(
        int(payload["feature_dim"]),
        payload["target_names"],
        dropout=(
            float(payload.get("dropout", 0.2)) if args.dropout is None else args.dropout
        ),
        attention_hidden_dim=(
            int(payload.get("attention_hidden_dim", 128))
            if args.attention_hidden_dim is None
            else args.attention_hidden_dim
        ),
    )
    model.load_state_dict(payload["model_state"], strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    features = torch.randn(
        args.batch_size,
        args.slices,
        int(payload["feature_dim"]),
        device=device,
    )
    mask = torch.ones(args.batch_size, args.slices, dtype=torch.bool, device=device)
    planes = torch.randint(0, 4, (args.batch_size, args.slices), device=device)
    with torch.inference_mode():
        for _ in range(10):
            model(features, mask, planes)
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(args.iterations):
            model(features, mask, planes)
        if device.type == "cuda":
            torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    parameters = sum(parameter.numel() for parameter in model.parameters())
    profile = {
        "checkpoint_bytes": Path(args.checkpoint).stat().st_size,
        "parameters": parameters,
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "batch_size": args.batch_size,
        "slices_per_study": args.slices,
        "latency_ms_per_study": 1000.0 * elapsed / (args.iterations * args.batch_size),
        "throughput_studies_per_second": args.iterations * args.batch_size / elapsed,
        "peak_cuda_memory_bytes": (
            int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else None
        ),
        "runtime": runtime_metadata(include_torch=True),
    }
    output = Path(args.output)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(profile, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
