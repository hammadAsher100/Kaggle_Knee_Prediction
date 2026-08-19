"""Unicode normalization and sentence/section parsing for radiology reports."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

WHITESPACE_PATTERN = re.compile(r"[\t\v\f\r ]+")
BLANK_LINE_PATTERN = re.compile(r"\n{2,}")
SENTENCE_BOUNDARY_PATTERN = re.compile(r"(?<=[.!?;:])\s+|\n+")
SECTION_PATTERN = re.compile(
    r"(?im)^(?P<header>[\wÀ-žΑ-ωА-яİıŞşĞğÇçÖöÜü' -]{2,40})\s*:\s*"
)


@dataclass(frozen=True)
class ReportSection:
    name: str
    text: str


@dataclass(frozen=True)
class ParsedReport:
    normalized_text: str
    sections: tuple[ReportSection, ...]
    sentences: tuple[str, ...]


def normalize_report(text: str, *, unicode_form: str = "NFKC") -> str:
    """Normalize Unicode and whitespace without translating or losing accents."""
    normalized = unicodedata.normalize(unicode_form, str(text)).replace("\x00", " ")
    normalized = "\n".join(
        WHITESPACE_PATTERN.sub(" ", line).strip() for line in normalized.splitlines()
    )
    return BLANK_LINE_PATTERN.sub("\n", normalized).strip()


def parse_sections(text: str) -> tuple[ReportSection, ...]:
    """Split explicit report headers while preserving otherwise unsectioned text."""
    matches = list(SECTION_PATTERN.finditer(text))
    if not matches:
        return (ReportSection("report", text),) if text else ()
    sections: list[ReportSection] = []
    prefix = text[: matches[0].start()].strip()
    if prefix:
        sections.append(ReportSection("preamble", prefix))
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        content = text[match.end() : end].strip()
        sections.append(ReportSection(match.group("header").casefold().strip(), content))
    return tuple(sections)


def split_sentences(text: str) -> tuple[str, ...]:
    """Create deterministic clause-sized units for local evidence classification."""
    return tuple(
        segment.strip()
        for segment in SENTENCE_BOUNDARY_PATTERN.split(text)
        if segment.strip()
    )


def parse_report(text: str, *, unicode_form: str = "NFKC") -> ParsedReport:
    normalized = normalize_report(text, unicode_form=unicode_form)
    sections = parse_sections(normalized)
    sentences = tuple(
        sentence
        for section in sections
        for sentence in split_sentences(section.text)
    )
    return ParsedReport(normalized, sections, sentences)
