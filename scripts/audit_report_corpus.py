"""Write privacy-preserving aggregate report language and length statistics."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.labeling.language_detection import detect_language
from src.labeling.report_parser import normalize_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report-column", default="Report")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reports = pd.read_csv(args.train_csv, usecols=[args.report_column])[args.report_column]
    missing = reports.isna()
    normalized = reports.fillna("").astype(str)
    detections = [detect_language(report) for report in normalized]
    lengths = normalized.str.len()
    canonical = normalized.map(lambda value: normalize_report(value).casefold())
    duplicate_counts = canonical[canonical.ne("")].value_counts()
    duplicate_groups = duplicate_counts[duplicate_counts > 1]
    payload = {
        "row_count": int(len(reports)),
        "missing_reports": int(missing.sum()),
        "language_counts": dict(Counter(item.language for item in detections)),
        "dominant_script_counts": dict(Counter(item.dominant_script for item in detections)),
        "language_confidence_mean": {
            language: float(
                sum(item.confidence for item in detections if item.language == language)
                / sum(item.language == language for item in detections)
            )
            for language in sorted({item.language for item in detections})
        },
        "length_quantiles": {
            str(quantile): float(value)
            for quantile, value in lengths.quantile(
                [0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1]
            ).items()
        },
        "exact_duplicate_reports": {
            "group_count": int(len(duplicate_groups)),
            "study_count": int(duplicate_groups.sum()),
            "maximum_group_size": (
                int(duplicate_groups.max()) if not duplicate_groups.empty else 0
            ),
        },
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
