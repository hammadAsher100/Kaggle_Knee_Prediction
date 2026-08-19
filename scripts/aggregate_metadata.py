"""Aggregate streamed DICOM parts into ordered series and study inventories."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.metadata_aggregation import aggregate_metadata
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
    parser.add_argument("--parts-dir", required=True)
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
    audit = aggregate_metadata(
        args.parts_dir,
        series_descriptors_path=competition_root / "train_series.csv",
        series_manifest_path=require_config_value(config, "paths.train_series_manifest"),
        study_inventory_path=require_config_value(config, "paths.train_study_inventory"),
        audit_path=require_config_value(config, "paths.train_metadata_audit"),
    )
    logger.info(
        "metadata_aggregation_complete",
        extra={
            "event": "metadata_aggregation_complete",
            "config_hash": config_hash(config),
            "environment_mode": environment_report.mode,
            "dicom_count": audit["dicom_count"],
            "study_count": audit["study_count"],
            "series_count": audit["series_count"],
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
