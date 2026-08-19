"""Deterministic offline language and writing-system detection for reports."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass

SCRIPT_NAMES: tuple[str, ...] = (
    "LATIN",
    "CYRILLIC",
    "GREEK",
    "ARABIC",
    "THAI",
    "HEBREW",
    "HANGUL",
    "HIRAGANA",
    "KATAKANA",
    "CJK",
)
DIRECT_SCRIPT_LANGUAGES = {
    "GREEK": "el",
    "ARABIC": "ar",
    "THAI": "th",
    "HEBREW": "he",
    "HANGUL": "ko",
    "HIRAGANA": "ja",
    "KATAKANA": "ja",
    "CJK": "zh",
}
LATIN_LANGUAGE_MARKERS: dict[str, frozenset[str]] = {
    "en": frozenset(
        {"the", "and", "of", "no", "without", "findings", "impression", "knee", "joint"}
    ),
    "es": frozenset(
        {"de", "la", "el", "sin", "hallazgos", "impresión", "rodilla", "articulación"}
    ),
    "fr": frozenset(
        {"de", "la", "le", "sans", "conclusion", "genou", "articulation", "épanchement"}
    ),
    "de": frozenset(
        {"der", "die", "das", "und", "kein", "keine", "beurteilung", "knie", "gelenk"}
    ),
    "it": frozenset(
        {"di", "la", "il", "senza", "conclusioni", "ginocchio", "articolazione"}
    ),
    "nl": frozenset(
        {"de", "het", "een", "en", "geen", "conclusie", "knie", "gewricht"}
    ),
    "tr": frozenset({"ve", "bir", "yok", "bulgular", "sonuç", "diz", "eklem"}),
    "pt": frozenset(
        {"de", "do", "da", "sem", "achados", "conclusão", "joelho", "articulação"}
    ),
    "hr": frozenset({"je", "bez", "nalaz", "zaključak", "koljeno", "zglob"}),
    "ro": frozenset({"și", "de", "fără", "concluzii", "genunchi", "articulație"}),
    "mt": frozenset({"tal", "għal", "mingħajr", "irkoppa", "ġog"}),
}
CYRILLIC_LANGUAGE_MARKERS: dict[str, frozenset[str]] = {
    "bg": frozenset({"на", "и", "без", "заключение", "коляно", "става"}),
    "mn": frozenset({"ба", "үгүй", "дүгнэлт", "өвдөг", "үе"}),
}
TOKEN_PATTERN = re.compile(r"[^\W\d_]+", flags=re.UNICODE)


@dataclass(frozen=True)
class LanguageDetection:
    language: str
    confidence: float
    dominant_script: str
    script_fraction: float
    marker_hits: int

    def to_dict(self) -> dict[str, str | float | int]:
        return asdict(self)


def character_script(character: str) -> str | None:
    """Return the Unicode writing system for one alphabetic character."""
    if not character.isalpha():
        return None
    name = unicodedata.name(character, "")
    return next((script for script in SCRIPT_NAMES if script in name), "OTHER")


def script_profile(text: str) -> dict[str, int]:
    """Count alphabetic characters by Unicode writing system."""
    return dict(Counter(script for char in text if (script := character_script(char))))


def _marker_language(
    tokens: set[str],
    markers: dict[str, frozenset[str]],
) -> tuple[str, int, int]:
    scores = {language: len(tokens.intersection(words)) for language, words in markers.items()}
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    best_language, best_score = ranked[0]
    runner_up = ranked[1][1] if len(ranked) > 1 else 0
    return best_language, best_score, runner_up


def detect_language(text: str) -> LanguageDetection:
    """Detect a report language offline, returning uncertainty explicitly."""
    profile = script_profile(text)
    alphabetic_count = sum(profile.values())
    if alphabetic_count == 0:
        return LanguageDetection("unknown", 0.0, "EMPTY", 0.0, 0)
    dominant_script, dominant_count = max(profile.items(), key=lambda item: item[1])
    script_fraction = dominant_count / alphabetic_count
    if dominant_script in DIRECT_SCRIPT_LANGUAGES:
        return LanguageDetection(
            DIRECT_SCRIPT_LANGUAGES[dominant_script],
            min(1.0, script_fraction),
            dominant_script,
            script_fraction,
            0,
        )

    normalized = unicodedata.normalize("NFKC", text).casefold()
    tokens = set(TOKEN_PATTERN.findall(normalized))
    if dominant_script == "CYRILLIC":
        language, score, runner_up = _marker_language(tokens, CYRILLIC_LANGUAGE_MARKERS)
    elif dominant_script == "LATIN":
        language, score, runner_up = _marker_language(tokens, LATIN_LANGUAGE_MARKERS)
    else:
        return LanguageDetection("unknown", 0.0, dominant_script, script_fraction, 0)

    if score < 2 or score == runner_up:
        return LanguageDetection("unknown", 0.0, dominant_script, script_fraction, score)
    margin = score - runner_up
    confidence = min(0.99, 0.45 + 0.08 * score + 0.07 * margin) * script_fraction
    return LanguageDetection(language, confidence, dominant_script, script_fraction, score)
