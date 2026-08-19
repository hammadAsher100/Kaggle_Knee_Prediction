"""Multilingual uncertainty classification for report evidence."""

from __future__ import annotations

UNCERTAINTY_MARKERS: dict[str, tuple[str, ...]] = {
    "en": ("possible", "possibly", "probable", "suspicious", "cannot exclude", "may represent"),
    "es": ("posible", "probable", "sugestivo", "no se descarta", "podría"),
    "fr": ("possible", "probable", "suspect", "ne peut exclure", "pourrait"),
    "de": ("möglich", "wahrscheinlich", "verdacht", "nicht auszuschließen"),
    "nl": ("mogelijk", "waarschijnlijk", "verdacht", "niet uit te sluiten"),
    "tr": ("olası", "olasi", "muhtemel", "şüpheli", "dışlanamaz"),
    "bg": ("възмож", "вероят", "съмнение", "не може да се изключи"),
    "el": ("πιθαν", "ύποπτ", "δεν μπορεί να αποκλειστεί"),
    "hr": ("moguć", "vjerojat", "suspekt", "ne može se isključiti"),
    "unknown": ("possible", "probable", "suspicious", "verdacht", "posible"),
}


def is_uncertain(context: str, language: str) -> bool:
    lowered = context.casefold()
    markers = UNCERTAINTY_MARKERS.get(language, ()) + UNCERTAINTY_MARKERS["unknown"]
    return any(marker in lowered for marker in markers)
