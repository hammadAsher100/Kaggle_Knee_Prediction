"""Stream train DICOM headers into resumable Parquet parts on Kaggle."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.metadata import extract_metadata_parts
from src.utils.config import load_config, require_config_value
from src.utils.environment import prepare_runtime_config
from src.utils.hashing import config_hash
from src.utils.logging_config import configure_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        action="append",
        default=["configs/base.yaml", "configs/data.yaml"],
        help="YAML config path; may be passed more than once",
    )
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--studies-per-part", type=int, default=25)
    parser.add_argument("--resume-manifest", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config, environment_report = prepare_runtime_config(load_config(args.config))
    logging_config = config.get("logging", {})
    logger = configure_logging(
        level=logging_config.get("level", "INFO"),
        log_file=logging_config.get("file"),
    )
    competition_root = Path(require_config_value(config, "paths.competition_root"))
    series_csv = competition_root / f"{args.split}_series.csv"
    series = pd.read_csv(series_csv, usecols=["StudyInstanceUID"])
    study_ids = sorted(series["StudyInstanceUID"].dropna().astype(str).unique())
    runtime = config["runtime"]
    result = extract_metadata_parts(
        require_config_value(config, f"paths.{args.split}_dicom_root"),
        study_ids,
        require_config_value(config, f"paths.{args.split}_metadata_parts_dir"),
        tags=config.get("metadata", {}).get("tags", []),
        manifest_path=require_config_value(config, f"paths.{args.split}_metadata_manifest"),
        failure_path=require_config_value(config, f"paths.{args.split}_metadata_failures"),
        resume_manifest=args.resume_manifest,
        studies_per_part=args.studies_per_part,
        max_runtime_seconds=float(runtime["max_seconds"]),
        safety_reserve_seconds=float(runtime["safety_reserve_seconds"]),
        logger=logger,
    )
    logger.info(
        "streaming_metadata_complete",
        extra={
            "event": "streaming_metadata_complete",
            "split": args.split,
            "config_hash": config_hash(config),
            "environment_mode": environment_report.mode,
            **result.to_dict(),
        },
    )
    return 0 if result.complete else 75


if __name__ == "__main__":
    raise SystemExit(main())
