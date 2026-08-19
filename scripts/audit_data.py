"""Audit competition table schemas without guessing unresolved fields."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.audit import audit_competition_tables, write_audit_json
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config, environment_report = prepare_runtime_config(load_config(args.config))
    logging_config = config.get("logging", {})
    logger = configure_logging(
        level=logging_config.get("level", "INFO"),
        log_file=logging_config.get("file"),
    )
    train_path = require_config_value(config, "paths.train_csv")
    sample_path = require_config_value(config, "paths.sample_submission_csv")
    output_path = require_config_value(config, "paths.table_audit_json")
    audit = audit_competition_tables(
        train_path,
        sample_path,
        configured_schema=config.get("schema"),
    )
    audit["config_hash"] = config_hash(config)
    output = write_audit_json(audit, output_path)
    logger.info(
        "table_audit_complete",
        extra={
            "event": "table_audit_complete",
            "output_path": str(output),
            "train_rows": audit["train_row_count"],
            "target_count": len(audit["schema"]["target_columns"]),
            "environment_mode": environment_report.mode,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
