"""Build a geometry-ordered test series manifest with a compact audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.inference_manifest import build_inference_series_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series-csv", required=True)
    parser.add_argument("--dicom-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--audit", required=True)
    args = parser.parse_args()
    manifest, audit = build_inference_series_manifest(
        pd.read_csv(args.series_csv),
        args.dicom_root,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    manifest.to_parquet(temporary, index=False)
    temporary.replace(output)
    audit_path = Path(args.audit)
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
