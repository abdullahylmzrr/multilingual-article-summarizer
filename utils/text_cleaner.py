"""Reusable text cleaning helpers for academic PDF content."""

from __future__ import annotations

import re


URL_PATTERN = re.compile(r"\b(?:https?://|www\.)\S+", flags=re.IGNORECASE)
EMAIL_PATTERN = re.compile(r"\b[\w.%+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
DOI_PATTERN = re.compile(
    r"\b(?:doi:\s*|https?://(?:dx\.)?doi\.org/)?10\.\d{4,9}/[-._;()/:A-Z0-9]+",
    flags=re.IGNORECASE,
)
PAGE_NUMBER_PATTERN = re.compile(
    r"^(?:page|sayfa)?\s*[-–—]?\s*\d+\s*(?:/\s*\d+)?\s*[-–—]?$",
    flags=re.IGNORECASE,
)

ENGLISH_REFERENCE_HEADINGS = ("References", "Bibliography", "Works Cited")
TURKISH_REFERENCE_HEADINGS = ("Kaynakça", "Kaynaklar", "KAYNAKÇA", "KAYNAKLAR")


def remove_urls(text: str) -> str:
    """Remove web URLs from text."""
    return URL_PATTERN.sub(" ", text)


def remove_emails(text: str) -> str:
    """Remove email addresses from text."""
    return EMAIL_PATTERN.sub(" ", text)


def remove_doi_patterns(text: str) -> str:
    """Remove DOI identifiers and DOI URLs from text."""
    return DOI_PATTERN.sub(" ", text)


def remove_page_numbers(text: str) -> str:
    """Remove standalone page-number lines from extracted PDF text."""
    cleaned_lines = []

    for line in text.splitlines():
        if PAGE_NUMBER_PATTERN.fullmatch(line.strip()):
            continue
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def remove_references_section(text: str, language: str) -> str:
    """Remove everything after a references section heading.

    Args:
        text: Input article text.
        language: Detected language code, usually ``"en"`` or ``"tr"``.

    Returns:
        Text before the first matching references heading.
    """
    normalized_language = language.lower().strip()
    if normalized_language == "en":
        headings = ENGLISH_REFERENCE_HEADINGS
    elif normalized_language == "tr":
        headings = TURKISH_REFERENCE_HEADINGS
    else:
        headings = ENGLISH_REFERENCE_HEADINGS + TURKISH_REFERENCE_HEADINGS

    escaped_headings = "|".join(re.escape(heading) for heading in headings)
    heading_pattern = re.compile(
        rf"^\s*(?:\d+\.?\s*)?(?:{escaped_headings})\s*:?\s*$",
        flags=re.IGNORECASE,
    )

    lines = text.splitlines()
    for index, line in enumerate(lines):
        if heading_pattern.match(line):
            return "\n".join(lines[:index]).strip()

    return text


def remove_extra_whitespace(text: str) -> str:
    """Normalize repeated whitespace while keeping paragraph breaks readable."""
    normalized_lines = [
        re.sub(r"[ \t]+", " ", line).strip()
        for line in text.splitlines()
    ]
    normalized_text = "\n".join(normalized_lines)
    normalized_text = re.sub(r"\n{3,}", "\n\n", normalized_text)
    normalized_text = re.sub(r"(?<!\n)\n(?!\n)", " ", normalized_text)
    normalized_text = re.sub(r"[ \t]+", " ", normalized_text)
    return normalized_text.strip()


def normalize_whitespace(text: str) -> str:
    """Backward-compatible alias for whitespace normalization."""
    return remove_extra_whitespace(text)
