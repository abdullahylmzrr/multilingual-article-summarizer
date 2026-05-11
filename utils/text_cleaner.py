"""Reusable text cleaning helpers for academic PDF content."""

from __future__ import annotations

import re


URL_PATTERN = re.compile(r"\b(?:https?://|www\.)\S+", flags=re.IGNORECASE)
EMAIL_PATTERN = re.compile(r"\b[\w.%+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
DOI_PATTERN = re.compile(
    r"\b(?:doi:\s*|https?://(?:dx\.)?doi\.org/)?10\.\d{4,9}/[-._;()/:A-Z0-9]+",
    flags=re.IGNORECASE,
)
WORD_FRAGMENT_PATTERN = re.compile(
    r"([A-Za-zÇĞİÖŞÜçğıöşü]+)-[\t ]*(?:\r?\n|\s)+([A-Za-zÇĞİÖŞÜçğıöşü]+)"
)
PAGE_NUMBER_PATTERN = re.compile(
    r"^(?:page|sayfa)?\s*[-–—]?\s*\d+\s*(?:/\s*\d+)?\s*[-–—]?$",
    flags=re.IGNORECASE,
)
URL_ONLY_PATTERN = re.compile(r"^\s*(?:https?://|www\.)\S+\s*$", flags=re.IGNORECASE)
MOSTLY_NUMBERS_OR_SYMBOLS_PATTERN = re.compile(r"^[\W\d_]+$", flags=re.UNICODE)
ACADEMIC_METADATA_PATTERNS = (
    re.compile(r"^\s*published by\b", flags=re.IGNORECASE),
    re.compile(r"^\s*copyright\b|^\s*©", flags=re.IGNORECASE),
    re.compile(r"^\s*journal of\b", flags=re.IGNORECASE),
    re.compile(r"^\s*vol\.?\s*\d+|^\s*volume\s+\d+", flags=re.IGNORECASE),
    re.compile(r"^\s*no\.?\s*\d+|^\s*number\s+\d+", flags=re.IGNORECASE),
    re.compile(r"\b(?:e-?issn|p-?issn|issn)\b", flags=re.IGNORECASE),
    re.compile(r"^\s*doi\b|\bdoi\.org\b|\b10\.\d{4,9}/", flags=re.IGNORECASE),
    re.compile(
        r"^\s*(?:received|accepted|revised|available online|published online)\b",
        flags=re.IGNORECASE,
    ),
)
AUTHOR_HEADER_PATTERN = re.compile(
    r"^[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'`.-]+(?:\s+[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'`.-]+){1,5}"
    r"(?:\s+\d+(?:,\d+)*)?$"
)
AFFILIATION_LINE_PATTERN = re.compile(
    r"\b(?:academy|centre|center|college|department|faculty|institute|laboratory|"
    r"lab|school|technical university|university)\b",
    flags=re.IGNORECASE,
)
ADDRESS_LINE_PATTERN = re.compile(
    r"\b(?:avenue|blvd|boulevard|city|denmark|italy|road|street|turkey|"
    r"united kingdom|united states|via)\b",
    flags=re.IGNORECASE,
)

ENGLISH_REFERENCE_HEADINGS = ("References", "REFERENCES", "Bibliography", "Works Cited")
TURKISH_REFERENCE_HEADINGS = ("Kaynakça", "KAYNAKÇA", "Kaynaklar", "KAYNAKLAR")


def remove_urls(text: str) -> str:
    """Remove web URLs from text."""
    return URL_PATTERN.sub(" ", text)


def remove_emails(text: str) -> str:
    """Remove email addresses from text."""
    return EMAIL_PATTERN.sub(" ", text)


def remove_doi_patterns(text: str) -> str:
    """Remove DOI identifiers and DOI URLs from text."""
    return DOI_PATTERN.sub(" ", text)


def fix_pdf_hyphenation(text: str) -> str:
    """Join words split by PDF line-break or spacing hyphenation.

    Examples:
        ``ko-\nrunmasına`` -> ``korunmasına``
        ``ko- Runmasına`` -> ``korunmasına``
        ``yo- lunda`` -> ``yolunda``
        ``mevzuat- la`` -> ``mevzuatla``
    """

    def join_fragments(match: re.Match[str]) -> str:
        left_fragment = match.group(1)
        right_fragment = match.group(2)

        if (
            left_fragment[-1:].islower()
            and right_fragment[:1].isupper()
            and any(character.islower() for character in right_fragment[1:])
        ):
            right_fragment = f"{right_fragment[0].lower()}{right_fragment[1:]}"

        return f"{left_fragment}{right_fragment}"

    return WORD_FRAGMENT_PATTERN.sub(join_fragments, text)


def remove_page_numbers(text: str) -> str:
    """Remove standalone page-number lines from extracted PDF text."""
    cleaned_lines = []

    for line in text.splitlines():
        if PAGE_NUMBER_PATTERN.fullmatch(line.strip()):
            continue
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def _normalize_line_for_counting(line: str) -> str:
    """Normalize a line for repeated header/footer detection."""
    lowered_line = line.lower().strip()
    lowered_line = re.sub(r"\d+", "#", lowered_line)
    return re.sub(r"\s+", " ", lowered_line)


def _is_mostly_numbers_or_symbols(line: str) -> bool:
    """Return whether a line contains little useful alphabetic text."""
    stripped_line = line.strip()
    if not stripped_line:
        return True

    if MOSTLY_NUMBERS_OR_SYMBOLS_PATTERN.fullmatch(stripped_line):
        return True

    alphabetic_count = sum(character.isalpha() for character in stripped_line)
    return alphabetic_count / max(len(stripped_line), 1) < 0.25


def _is_metadata_line(line: str) -> bool:
    """Return whether a line looks like academic PDF metadata."""
    stripped_line = line.strip()
    if not stripped_line:
        return True

    if EMAIL_PATTERN.search(stripped_line) or URL_ONLY_PATTERN.fullmatch(stripped_line):
        return True

    return any(pattern.search(stripped_line) for pattern in ACADEMIC_METADATA_PATTERNS)


def _looks_like_affiliation_line(line: str) -> bool:
    """Return whether a line looks like an author affiliation or address."""
    comma_count = line.count(",")
    digit_count = sum(character.isdigit() for character in line)
    starts_with_index = bool(re.match(r"^\s*\d+\s+", line))
    has_affiliation = bool(AFFILIATION_LINE_PATTERN.search(line))
    has_address = bool(ADDRESS_LINE_PATTERN.search(line))

    if starts_with_index and has_affiliation:
        return True
    if has_affiliation and has_address and comma_count >= 1:
        return True
    if has_affiliation and comma_count >= 2 and digit_count >= 1:
        return True

    return False


def _looks_like_repeated_author_header(line: str, line_counts: dict[str, int]) -> bool:
    """Return whether a line is likely a repeated author/header line."""
    normalized_line = _normalize_line_for_counting(line)
    if line_counts.get(normalized_line, 0) < 3:
        return False

    words = line.split()
    if len(words) > 14:
        return False

    return bool(AUTHOR_HEADER_PATTERN.match(line.strip())) or len(line) <= 120


def _looks_like_author_line(line: str) -> bool:
    """Return whether a line looks like a standalone author-name line."""
    stripped_line = line.strip()
    if stripped_line.endswith((".", "!", "?")):
        return False

    words = stripped_line.split()
    if not 2 <= len(words) <= 14:
        return False

    name_like_tokens = [
        word
        for word in words
        if re.match(r"^[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'`.-]+(?:\d+|,\d+)*$", word)
    ]
    digit_count = sum(character.isdigit() for character in stripped_line)

    if len(name_like_tokens) >= 3 and digit_count > 0:
        return True
    if len(name_like_tokens) >= 4 and len(name_like_tokens) / len(words) > 0.7:
        return True

    return False


def _drop_front_matter_before_abstract(lines: list[str]) -> list[str]:
    """Drop title/author front matter when an abstract heading is found."""
    for index, line in enumerate(lines):
        if re.match(r"^\s*(?:abstract|öz|özet)\b", line, flags=re.IGNORECASE):
            return lines[index:]

    return lines


def remove_academic_pdf_noise(text: str) -> str:
    """Remove common academic PDF header, footer, and metadata noise lines."""
    lines = _drop_front_matter_before_abstract(text.splitlines())
    line_counts: dict[str, int] = {}

    for line in lines:
        normalized_line = _normalize_line_for_counting(line)
        if normalized_line:
            line_counts[normalized_line] = line_counts.get(normalized_line, 0) + 1

    useful_lines = []
    for line in lines:
        stripped_line = line.strip()
        if not stripped_line:
            useful_lines.append("")
            continue

        if PAGE_NUMBER_PATTERN.fullmatch(stripped_line):
            continue
        if _is_metadata_line(stripped_line):
            continue
        if _looks_like_affiliation_line(stripped_line):
            continue
        if _looks_like_author_line(stripped_line):
            continue
        if _looks_like_repeated_author_header(stripped_line, line_counts):
            continue

        useful_lines.append(line)

    return "\n".join(useful_lines)


def clean_lines(text: str) -> str:
    """Remove empty, noisy, metadata-only, and very short lines."""
    useful_lines = []

    for line in text.splitlines():
        stripped_line = line.strip()
        if not stripped_line or len(stripped_line) < 3:
            continue
        if _is_mostly_numbers_or_symbols(stripped_line):
            continue
        if _is_metadata_line(stripped_line):
            continue
        if _looks_like_affiliation_line(stripped_line):
            continue
        if _looks_like_author_line(stripped_line):
            continue

        useful_lines.append(stripped_line)

    return "\n".join(useful_lines)


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
        rf"^\s*(?:\d+(?:\.\d+)*\.?\s+)?(?:{escaped_headings})\s*:?\s*$",
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
