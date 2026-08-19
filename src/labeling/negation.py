"""Multilingual local-context negation detection."""

from __future__ import annotations

NEGATION_MARKERS: dict[str, tuple[str, ...]] = {
    "en": ("no ", "not ", "without ", "absence of ", "negative for ", "intact", "normal"),
    "es": ("no ", "sin ", "ausencia de ", "íntegro", "integro", "normal"),
    "fr": ("pas de ", "sans ", "absence de ", "intact", "normal"),
    "de": ("kein ", "keine ", "keinen ", "ohne ", "intakt", "unauffällig"),
    "nl": ("geen ", "zonder ", "intact", "normaal"),
    "tr": ("yok", "izlenmedi", "saptanmadı", "saptanmadi", "intakt", "normal"),
    "bg": ("няма", "без ", "не се установява", "интакт", "нормал"),
    "el": ("χωρίς", "δεν ", "απουσία", "ακέραι", "φυσιολογ"),
    "hr": ("nema", "bez ", "nije", "uredan", "intaktan", "normalan"),
    "unknown": ("no ", "sin ", "sans ", "kein ", "geen ", "yok", "bez ", "χωρίς"),
}


def is_negated(context_before: str, context_after: str, language: str) -> bool:
    """Return whether a nearby target mention is explicitly absent or intact."""
    before = context_before.casefold()[-100:]
    after = context_after.casefold()[:60]
    markers = NEGATION_MARKERS.get(language, ()) + NEGATION_MARKERS["unknown"]
    return any(marker in before for marker in markers) or any(
        marker.strip() in after for marker in markers if marker.strip() in {"intact", "normal"}
    )
