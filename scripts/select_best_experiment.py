"""Select and promote the best measured gold-only OOF experiment."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    root = Path(args.experiments_dir).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    candidates: list[tuple[float, str, Path]] = []
    for metrics_path in sorted(root.glob("*/oof_metrics.json")):
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        score = payload.get("macro_auc")
        if score is not None and isinstance(score, (int, float)):
            candidates.append((float(score), metrics_path.parent.name, metrics_path.parent))
    if not candidates:
        raise RuntimeError("No completed experiment metrics were found")
    score, experiment_id, source = max(candidates, key=lambda item: (item[0], item[1]))
    output.mkdir(parents=True, exist_ok=True)
    promoted_files = [
        *source.glob("fold-*.pt"),
        source / "oof.parquet",
        source / "oof_metrics.json",
    ]
    for source_file in promoted_files:
        if source_file.is_file():
            shutil.copy2(source_file, output / source_file.name)
    summary = {
        "selected_experiment": experiment_id,
        "selected_macro_auc": score,
        "candidate_scores": {
            name: candidate_score for candidate_score, name, _ in candidates
        },
    }
    (output / "best_experiment.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
