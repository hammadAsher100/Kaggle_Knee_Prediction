"""Tests for deterministic offline report language detection."""

from __future__ import annotations

from src.labeling.language_detection import detect_language, script_profile


def test_script_profile_detects_non_latin_writing_systems() -> None:
    profile = script_profile("knee γόνατο الركبة เข่า")

    assert profile["LATIN"] == 4
    assert profile["GREEK"] > 0
    assert profile["ARABIC"] > 0
    assert profile["THAI"] > 0


def test_language_detection_uses_script_and_marker_evidence() -> None:
    english = detect_language("Findings of the knee joint and no fracture")
    spanish = detect_language("Hallazgos de la rodilla y articulación sin fractura")
    greek = detect_language("Μαγνητική τομογραφία γόνατος")

    assert english.language == "en"
    assert spanish.language == "es"
    assert greek.language == "el"
    assert english.confidence > 0


def test_language_detection_reports_ambiguous_text_as_unknown() -> None:
    detection = detect_language("ACL MCL MRI")

    assert detection.language == "unknown"
    assert detection.marker_hits == 0
