"""Validate local or Kaggle execution configuration without scanning DICOM files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import load_config
from src.utils.environment import prepare_runtime_config
from src.utils.hashing import config_hash
from src.utils.logging_config import configure_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        action="append",
        default=["configs/base.yaml", "configs/data.yaml", "configs/local.yaml"],
        help="YAML config path; later files override earlier files",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw: dict[str, Any] = load_config(args.config)
    config, report = prepare_runtime_config(raw)
    logger = configure_logging(level=config.get("logging", {}).get("level", "INFO"))
    logger.info(
        "environment_validated",
        extra={
            "event": "environment_validated",
            "config_hash": config_hash(config),
            **report.to_dict(),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
