"""Extract resumable per-study frozen DINOv2 embeddings from selected MRI slices."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import KneeStudyDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-table", required=True)
    parser.add_argument("--series-manifest", required=True)
    parser.add_argument("--dicom-root", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--slices-per-series", type=int, default=4)
    parser.add_argument("--max-series", type=int, default=3)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--studies-per-part", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-runtime-seconds", type=float, default=32400)
    parser.add_argument("--safety-reserve-seconds", type=float, default=1200)
    parser.add_argument("--minimum-valid-stack-fraction", type=float, default=0.9)
    return parser.parse_args()


def _read_table(path: str) -> pd.DataFrame:
    return pd.read_csv(path) if Path(path).suffix.lower() == ".csv" else pd.read_parquet(path)


def _write_npz(path: Path, **arrays) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    if not 0 <= args.minimum_valid_stack_fraction <= 1:
        raise ValueError("minimum valid stack fraction must lie in [0, 1]")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import AutoModel

    study_table = _read_table(args.study_table)
    if "StudyInstanceUID" not in study_table:
        raise ValueError("study table is missing StudyInstanceUID")
    studies = study_table[["StudyInstanceUID"]].copy()
    manifest = pd.read_parquet(args.series_manifest)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained(args.model_path, local_files_only=True).to(device).eval()
    feature_dim = int(model.config.hidden_size)
    started = time.monotonic()
    part_paths: list[str] = []
    completed = 0
    valid_stacks = 0
    total_stacks = 0
    for part_index, start in enumerate(range(0, len(studies), args.studies_per_part)):
        part_path = output / f"part-{part_index:05d}.npz"
        part_paths.append(part_path.name)
        if part_path.is_file():
            with np.load(part_path, allow_pickle=False) as existing:
                completed += len(existing["study_ids"])
                valid_stacks += int(existing["slice_mask"].sum())
                total_stacks += int(existing["slice_mask"].size)
            continue
        elapsed = time.monotonic() - started
        if elapsed >= args.max_runtime_seconds - args.safety_reserve_seconds:
            break
        subset = studies.iloc[start : start + args.studies_per_part]
        dataset = KneeStudyDataset(
            subset,
            manifest,
            args.dicom_root,
            slices_per_series=args.slices_per_series,
            max_series=args.max_series,
            image_size=args.image_size,
            training=False,
        )
        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        ids: list[str] = []
        features: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        planes: list[np.ndarray] = []
        with torch.inference_mode():
            for batch in loader:
                images = batch["images"].squeeze(0).to(device, non_blocking=True)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=device.type == "cuda",
                ):
                    encoded = model(pixel_values=images).last_hidden_state[:, 0]
                valid = batch["slice_mask"].squeeze(0).numpy().astype(bool)
                values = encoded.float().cpu().numpy()
                values[~valid] = 0.0
                ids.append(str(batch["study_id"][0]))
                features.append(values.astype(np.float16))
                masks.append(valid)
                planes.append(batch["plane_ids"].squeeze(0).numpy().astype(np.int8))
        _write_npz(
            part_path,
            study_ids=np.asarray(ids, dtype="U80"),
            features=np.stack(features),
            slice_mask=np.stack(masks),
            plane_ids=np.stack(planes),
        )
        completed += len(ids)
        valid_stacks += int(np.stack(masks).sum())
        total_stacks += int(np.stack(masks).size)
        print(
            json.dumps(
                {
                    "part": part_index,
                    "completed_studies": completed,
                    "total_studies": len(studies),
                    "elapsed_seconds": time.monotonic() - started,
                }
            ),
            flush=True,
        )
    complete = completed == len(studies)
    manifest_payload = {
        "complete": complete,
        "completed_studies": completed,
        "total_studies": len(studies),
        "feature_dim": feature_dim,
        "slices_per_study": args.slices_per_series * args.max_series,
        "parts": part_paths[: (completed + args.studies_per_part - 1) // args.studies_per_part],
        "valid_stack_fraction": valid_stacks / max(total_stacks, 1),
    }
    manifest_path = output / "manifest.json"
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(json.dumps(manifest_payload, indent=2) + "\n", encoding="utf-8")
    temporary_manifest.replace(manifest_path)
    if not complete:
        return 75
    if manifest_payload["valid_stack_fraction"] < args.minimum_valid_stack_fraction:
        raise RuntimeError(
            "More than 10% of selected 2.5D stacks failed DICOM pixel decoding; "
            "check installed transfer-syntax handlers before training"
        )
    archives = [np.load(output / name, allow_pickle=False) for name in manifest_payload["parts"]]
    try:
        _write_npz(
            output / "features.npz",
            study_ids=np.concatenate([archive["study_ids"] for archive in archives]),
            features=np.concatenate([archive["features"] for archive in archives]),
            slice_mask=np.concatenate([archive["slice_mask"] for archive in archives]),
            plane_ids=np.concatenate([archive["plane_ids"] for archive in archives]),
        )
    finally:
        for archive in archives:
            archive.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
