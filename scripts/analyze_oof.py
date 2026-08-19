"""Create uncertainty and sex-subgroup audits from immutable OOF predictions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.oof_analysis import bootstrap_macro_auc, subgroup_auc

TARGETS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oof", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    args = parser.parse_args()
    frame = pd.read_parquet(args.oof)
    truth = frame[[f"{target}__gold" for target in TARGETS]].to_numpy(float)
    probabilities = frame[[f"{target}__prediction" for target in TARGETS]].to_numpy(float)
    audit = {
        "gold_macro_auc_bootstrap": bootstrap_macro_auc(
            truth,
            probabilities,
            TARGETS,
            iterations=args.bootstrap_iterations,
        ),
        "patient_sex": subgroup_auc(frame, TARGETS, subgroup_column="PatientSex"),
        "limitations": [
            "Only 58 gold studies per target are available.",
            "Subgroup estimates may be undefined or high variance when a class is absent.",
            "Weak-label performance is not substituted for gold-label performance.",
        ],
    }
    output = Path(args.output)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(audit, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
