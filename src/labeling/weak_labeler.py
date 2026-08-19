"""Deterministic multilingual report weak-label generation."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from functools import cache

from src.labeling.language_detection import LanguageDetection, detect_language
from src.labeling.negation import is_negated
from src.labeling.report_parser import parse_report
from src.labeling.terminology import TARGET_TERMINOLOGY, pathology_markers
from src.labeling.uncertainty import is_uncertain


@dataclass(frozen=True)
class WeakTargetLabel:
    target: str
    probability: float
    binary_label: int | None
    confidence: float
    status: str
    evidence: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class WeakReportLabels:
    language: LanguageDetection
    targets: dict[str, WeakTargetLabel]


@cache
def _term_pattern(term: str) -> re.Pattern[str]:
    return re.compile(rf"(?<!\w){re.escape(term.casefold())}(?!\w)", flags=re.UNICODE)


def _sentence_evidence(
    sentence: str,
    *,
    language: str,
    terms: tuple[str, ...],
    requires_pathology_marker: bool,
) -> tuple[list[str], list[str], list[str], list[str]]:
    lowered = sentence.casefold()
    positive: list[str] = []
    negative: list[str] = []
    uncertain: list[str] = []
    mentions: list[str] = []
    has_pathology = any(marker in lowered for marker in pathology_markers(language))
    for term in terms:
        for match in _term_pattern(term).finditer(lowered):
            before = lowered[: match.start()]
            after = lowered[match.end() :]
            mentions.append(sentence)
            if is_negated(before, after, language):
                negative.append(sentence)
            elif is_uncertain(sentence, language):
                uncertain.append(sentence)
            elif requires_pathology_marker and not has_pathology:
                continue
            else:
                positive.append(sentence)
    return positive, negative, uncertain, mentions


def label_target(
    sentences: tuple[str, ...],
    *,
    target: str,
    language: str,
) -> WeakTargetLabel:
    if target not in TARGET_TERMINOLOGY:
        raise KeyError(f"No terminology configured for target: {target}")
    specification = TARGET_TERMINOLOGY[target]
    positive: list[str] = []
    negative: list[str] = []
    uncertain: list[str] = []
    mentions: list[str] = []
    for sentence in sentences:
        evidence = _sentence_evidence(
            sentence,
            language=language,
            terms=specification.terms,
            requires_pathology_marker=specification.requires_pathology_marker,
        )
        positive.extend(evidence[0])
        negative.extend(evidence[1])
        uncertain.extend(evidence[2])
        mentions.extend(evidence[3])

    if positive:
        return WeakTargetLabel(target, 0.95, 1, 0.9, "positive", tuple(dict.fromkeys(positive)))
    if uncertain:
        return WeakTargetLabel(
            target,
            0.5,
            None,
            0.25,
            "uncertain",
            tuple(dict.fromkeys(uncertain)),
        )
    if negative:
        return WeakTargetLabel(target, 0.05, 0, 0.85, "negative", tuple(dict.fromkeys(negative)))
    if mentions:
        return WeakTargetLabel(
            target,
            0.5,
            None,
            0.15,
            "mentioned_without_status",
            tuple(dict.fromkeys(mentions)),
        )
    return WeakTargetLabel(target, 0.5, None, 0.0, "not_mentioned", ())


def label_report(
    text: str,
    *,
    target_columns: tuple[str, ...],
    unicode_form: str = "NFKC",
) -> WeakReportLabels:
    parsed = parse_report(text, unicode_form=unicode_form)
    language = detect_language(parsed.normalized_text)
    labels = {
        target: label_target(
            parsed.sentences,
            target=target,
            language=language.language,
        )
        for target in target_columns
    }
    return WeakReportLabels(language, labels)
