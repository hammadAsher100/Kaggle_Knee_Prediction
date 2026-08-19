"""Generate and audit deterministic leakage-safe multilabel folds."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.audit import read_table
from src.training.cv_split import (
    assert_train_valid_disjoint,
    fold_audit,
    make_multilabel_group_folds,
)
from src.utils.config import load_config, require_config_value
from src.utils.environment import prepare_runtime_config
from src.utils.logging_config import configure_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        action="append",
        default=["configs/base.yaml", "configs/data.yaml", "configs/cv.yaml"],
        help="YAML config path; may be passed more than once",
    )
    return parser.parse_args()


def _write_frame(frame: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        frame.to_csv(path, index=False)
    elif path.suffix.lower() in {".parquet", ".pq"}:
        frame.to_parquet(path, index=False)
    else:
        raise ValueError("Fold output must use .csv, .parquet, or .pq")


def main() -> int:
    args = parse_args()
    config, environment_report = prepare_runtime_config(load_config(args.config))
    logger = configure_logging(level=config.get("logging", {}).get("level", "INFO"))
    train = read_table(require_config_value(config, "paths.train_csv"))
    schema = config["schema"]
    study_column = require_config_value(config, "schema.study_identifier")
    targets = require_config_value(config, "schema.target_columns")
    patient_column = schema.get("patient_identifier")
    group_column = patient_column or study_column
    configured_group = config["cv"].get("grouping_key", "auto")
    if configured_group != "auto":
        group_column = configured_group

    folded, quality = make_multilabel_group_folds(
        train,
        group_column=group_column,
        target_columns=targets,
        n_splits=int(config["cv"]["n_splits"]),
        seed=int(config["cv"]["seed"]),
        restarts=int(config["cv"].get("restarts", 64)),
    )
    for fold in range(int(config["cv"]["n_splits"])):
        assert_train_valid_disjoint(
            folded,
            validation_fold=fold,
            study_column=study_column,
            patient_column=patient_column,
        )
    audit = fold_audit(
        folded,
        target_columns=targets,
        study_column=study_column,
        patient_column=patient_column,
        subgroup_columns=config["cv"].get("subgroup_columns", []),
    )
    audit["quality"] = quality.to_dict()
    audit["grouping_key"] = group_column

    output = Path(config["cv"]["output_path"]).expanduser().resolve()
    audit_output = Path(config["cv"]["audit_path"]).expanduser().resolve()
    _write_frame(folded, output)
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    temporary = audit_output.with_suffix(audit_output.suffix + ".tmp")
    temporary.write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(audit_output)
    logger.info(
        "folds_built",
        extra={
            "event": "folds_built",
            "grouping_key": group_column,
            "output_path": str(output),
            "audit_path": str(audit_output),
            "objective": quality.objective,
            "environment_mode": environment_report.mode,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
