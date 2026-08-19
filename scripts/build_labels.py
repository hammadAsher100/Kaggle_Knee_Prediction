"""Build versioned multilingual report weak labels with gold validation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.labeling.label_audit import audit_weak_labels
from src.labeling.weak_labeler import label_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--config", default="configs/labeler.yaml")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))["labeler"]
    targets = tuple(config["target_columns"])
    train = pd.read_csv(args.train_csv)
    required = {"StudyInstanceUID", "Report", *targets}
    missing = sorted(required.difference(train.columns))
    if missing:
        raise ValueError(f"Training CSV is missing columns: {', '.join(missing)}")

    records: list[dict[str, object]] = []
    for row_values in train.to_dict(orient="records"):
        labels = label_report(
            str(row_values["Report"]),
            target_columns=targets,
            unicode_form=str(config["unicode_normalization"]),
        )
        record: dict[str, object] = {
            "StudyInstanceUID": str(row_values["StudyInstanceUID"]),
            "language": labels.language.language,
            "language_confidence": labels.language.confidence,
            "dominant_script": labels.language.dominant_script,
        }
        for target in targets:
            result = labels.targets[target]
            gold = row_values[target]
            gold_known = pd.notna(gold)
            record[f"{target}__gold"] = int(gold) if gold_known else None
            record[f"{target}__weak_probability"] = result.probability
            record[f"{target}__weak_label"] = result.binary_label
            record[f"{target}__weak_confidence"] = result.confidence
            record[f"{target}__weak_status"] = result.status
            record[f"{target}__evidence"] = json.dumps(
                result.evidence,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            record[f"{target}__probability"] = float(gold) if gold_known else result.probability
            record[f"{target}__label"] = int(gold) if gold_known else result.binary_label
            record[f"{target}__confidence"] = 1.0 if gold_known else result.confidence
            record[f"{target}__source"] = "gold" if gold_known else "rules_v1"
        records.append(record)

    output = Path(args.output or config["output_path"]).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    labels_frame = pd.DataFrame.from_records(records)
    temporary = output.with_suffix(output.suffix + ".tmp")
    labels_frame.to_parquet(temporary, index=False)
    temporary.replace(output)

    audit = audit_weak_labels(labels_frame, target_columns=targets)
    audit["labeler_version"] = config["version"]
    audit_output = output.with_suffix(".audit.json")
    temporary_audit = audit_output.with_suffix(audit_output.suffix + ".tmp")
    temporary_audit.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary_audit.replace(audit_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
