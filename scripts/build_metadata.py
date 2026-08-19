"""Extract train/test DICOM headers without decoding pixel arrays."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.metadata import build_study_inventory, extract_metadata, metadata_audit
from src.utils.config import load_config, require_config_value
from src.utils.environment import prepare_runtime_config
from src.utils.logging_config import configure_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        action="append",
        default=["configs/base.yaml", "configs/data.yaml"],
        help="YAML config path; may be passed more than once",
    )
    parser.add_argument("--split", choices=("train", "test", "both"), default="both")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config, environment_report = prepare_runtime_config(load_config(args.config))
    logging_config = config.get("logging", {})
    logger = configure_logging(
        level=logging_config.get("level", "INFO"),
        log_file=logging_config.get("file"),
    )
    tags = config.get("metadata", {}).get("tags", [])
    splits = ("train", "test") if args.split == "both" else (args.split,)
    for split in splits:
        root = require_config_value(config, f"paths.{split}_dicom_root")
        output = require_config_value(config, f"paths.{split}_metadata_output")
        failures = require_config_value(config, f"paths.{split}_metadata_failures")
        extract_metadata(
            root,
            output,
            tags=tags,
            failure_path=failures,
            logger=logger,
        )
        output_path = Path(output).expanduser().resolve()
        if output_path.suffix.lower() == ".csv":
            metadata = pd.read_csv(output_path)
        else:
            metadata = pd.read_parquet(output_path)
        inventory = build_study_inventory(metadata)
        inventory_path = Path(
            require_config_value(config, f"paths.{split}_study_inventory")
        ).expanduser().resolve()
        inventory_path.parent.mkdir(parents=True, exist_ok=True)
        if inventory_path.suffix.lower() == ".csv":
            inventory.to_csv(inventory_path, index=False)
        elif inventory_path.suffix.lower() in {".parquet", ".pq"}:
            inventory.to_parquet(inventory_path, index=False)
        else:
            raise ValueError("Study inventory output must use .csv, .parquet, or .pq")

        audit_path = Path(
            require_config_value(config, f"paths.{split}_metadata_audit")
        ).expanduser().resolve()
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = audit_path.with_suffix(audit_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(metadata_audit(metadata), indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(audit_path)
        logger.info(
            "dicom_audit_complete",
            extra={
                "event": "dicom_audit_complete",
                "split": split,
                "inventory_path": str(inventory_path),
                "audit_path": str(audit_path),
                "environment_mode": environment_report.mode,
            },
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
