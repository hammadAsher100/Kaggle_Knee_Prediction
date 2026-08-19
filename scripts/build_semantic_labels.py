"""Build offline multilingual semantic report labels and gold-label blend audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.labeling.report_parser import parse_report
from src.labeling.semantic_labeler import (
    blend_probabilities,
    encode_semantic_probabilities,
    stable_model_identifier,
)
from src.labeling.weak_labeler import label_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--config", default="configs/labeler.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def _macro_auc(
    gold: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[float, dict[int, float]]:
    per_target: dict[int, float] = {}
    for target_index in range(gold.shape[1]):
        known = np.isfinite(gold[:, target_index])
        values = gold[known, target_index]
        if len(np.unique(values)) == 2:
            per_target[target_index] = float(
                roc_auc_score(values.astype(int), probabilities[known, target_index])
            )
    return float(sum(per_target.values()) / len(per_target)), per_target


def main() -> int:
    args = parse_args()
    from sentence_transformers import SentenceTransformer

    labeler_config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))["labeler"]
    targets = tuple(labeler_config["target_columns"])
    train = pd.read_csv(args.train_csv)
    reports = train["Report"].fillna("").astype(str).tolist()
    parsed = [
        parse_report(report, unicode_form=labeler_config["unicode_normalization"])
        for report in reports
    ]
    report_sentences = [item.sentences for item in parsed]

    model = SentenceTransformer(args.model_path, device="cuda")
    semantic, positive_scores, negative_scores = encode_semantic_probabilities(
        model,
        report_sentences,
        target_columns=targets,
        batch_size=args.batch_size,
    )

    rule_probabilities = np.empty_like(semantic)
    languages: list[str] = []
    for row_index, report in enumerate(reports):
        rules = label_report(report, target_columns=targets)
        languages.append(rules.language.language)
        for target_index, target in enumerate(targets):
            rule_probabilities[row_index, target_index] = rules.targets[target].probability

    gold = train.loc[:, targets].to_numpy(dtype=np.float64)
    candidate_weights = (0.0, 0.25, 0.5, 0.75, 1.0)
    weight_metrics: dict[str, float] = {}
    blended_candidates: dict[float, np.ndarray] = {}
    for weight in candidate_weights:
        candidate = blend_probabilities(
            rule_probabilities,
            semantic,
            rule_weight=weight,
        )
        blended_candidates[weight] = candidate
        weight_metrics[str(weight)] = _macro_auc(gold, candidate)[0]
    selected_weight = max(candidate_weights, key=lambda value: (weight_metrics[str(value)], -value))
    selected = blended_candidates[selected_weight]
    selected_macro, selected_per_target = _macro_auc(gold, selected)

    output_frame = pd.DataFrame(
        {
            "StudyInstanceUID": train["StudyInstanceUID"].astype(str),
            "language": languages,
        }
    )
    for target_index, target in enumerate(targets):
        output_frame[f"{target}__semantic_probability"] = semantic[:, target_index]
        output_frame[f"{target}__rule_probability"] = rule_probabilities[:, target_index]
        output_frame[f"{target}__probability"] = selected[:, target_index]
        output_frame[f"{target}__positive_similarity"] = positive_scores[:, target_index]
        output_frame[f"{target}__negative_similarity"] = negative_scores[:, target_index]
        output_frame[f"{target}__gold"] = gold[:, target_index]

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    output_frame.to_parquet(temporary, index=False)
    temporary.replace(output)
    audit = {
        "row_count": int(len(output_frame)),
        "model": stable_model_identifier(args.model_path),
        "gold_studies": int(np.isfinite(gold).all(axis=1).sum()),
        "candidate_rule_weights": weight_metrics,
        "selected_rule_weight": selected_weight,
        "selected_macro_gold_auc": selected_macro,
        "selected_per_target_gold_auc": {
            targets[index]: value for index, value in selected_per_target.items()
        },
    }
    audit_path = output.with_suffix(".audit.json")
    temporary_audit = audit_path.with_suffix(audit_path.suffix + ".tmp")
    temporary_audit.write_text(
        json.dumps(audit, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary_audit.replace(audit_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
