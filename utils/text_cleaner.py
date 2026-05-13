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


def _fix_pdf_hyphenation_with_count(text: str) -> tuple[str, int]:
    """Join PDF-split word fragments and return the number of repairs."""
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

    return WORD_FRAGMENT_PATTERN.subn(join_fragments, text)


def fix_pdf_hyphenation(text: str) -> str:
    """Join words split by PDF line-break or spacing hyphenation.

    Examples:
        ``ko-\nrunmasına`` -> ``korunmasına``
        ``ko- Runmasına`` -> ``korunmasına``
        ``yo- lunda`` -> ``yolunda``
        ``mevzuat- la`` -> ``mevzuatla``
    """
    repaired_text, _ = _fix_pdf_hyphenation_with_count(text)
    return repaired_text


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


# ---------------------------------------------------------------------------
# Summarization-focused cleaning helpers
# ---------------------------------------------------------------------------

_TURKISH_METADATA_PATTERNS = (
    re.compile(r"\b(?:cilt|sayı|sayfa)\s*[:/]?\s*\d+", flags=re.IGNORECASE),
    re.compile(r"\bgeliş\s+tarihi\b", flags=re.IGNORECASE),
    re.compile(r"\bkabul\s+tarihi\b", flags=re.IGNORECASE),
    re.compile(r"\bsorumlu\s+yazar\b", flags=re.IGNORECASE),
    re.compile(r"\banahtar\s+kelimeler\b\s*:?", flags=re.IGNORECASE),
)

_ENGLISH_METADATA_PATTERNS = (
    re.compile(r"\b(?:volume|vol\.?|issue|no\.?|pages?)\s*[:/]?\s*\d+", flags=re.IGNORECASE),
    re.compile(r"^\s*(?:received|accepted|revised|available\s+online)\b", flags=re.IGNORECASE),
    re.compile(r"\bcorresponding\s+author\b", flags=re.IGNORECASE),
    re.compile(r"\bkeywords\b\s*:?", flags=re.IGNORECASE),
    re.compile(r"^\s*published\s+by\b", flags=re.IGNORECASE),
    re.compile(r"^\s*copyright\b|^\s*©", flags=re.IGNORECASE),
)

_SHARED_METADATA_PATTERNS = (
    re.compile(r"\b(?:e-?issn|p-?issn|issn)\b", flags=re.IGNORECASE),
    re.compile(r"\bdoi\s*:|\bdoi\.org\b|\b10\.\d{4,9}/", flags=re.IGNORECASE),
)

_CURRENCY_PATTERN = re.compile(r"[€$£₺]|\b(?:TL|TRY|USD|EUR|Euro)\b", flags=re.IGNORECASE)
_NUMBER_TOKEN_PATTERN = re.compile(r"^[+-]?\d+(?:[.,:/-]\d+)*%?$")
_DECIMAL_OR_AMOUNT_PATTERN = re.compile(r"\d+[.,]\d+|\d+\s*(?:%|€|\$|£|₺|\bTL\b|\bEuro\b)", flags=re.IGNORECASE)
_TABLE_HEADER_TERM_PATTERN = re.compile(
    r"\b(?:adet|tutar\w*|toplam|oran|yüzde|type|count|amount|total|rate|percent)\b",
    flags=re.IGNORECASE,
)
_COMMON_SECTION_HEADINGS = {
    "abstract",
    "acknowledgements",
    "conclusion",
    "discussion",
    "introduction",
    "method",
    "methodology",
    "results",
    "özet",
    "giriş",
    "sonuç",
    "sonuçlar",
    "tartışma",
    "yöntem",
}


def _count_removed_lines(original_lines: list[str], kept_lines: list[str]) -> int:
    """Return a conservative removed-line count."""
    return max(len(original_lines) - len(kept_lines), 0)


def _line_has_sentence_structure(line: str) -> bool:
    """Return whether a line has enough shape to look like prose."""
    stripped = line.strip()
    words = stripped.split()
    if len(words) >= 12 and stripped.endswith((".", "!", "?", ".”", ".’")):
        return True
    return len(words) >= 18 and stripped.count(",") <= 4


def _metadata_patterns_for_language(language: str) -> tuple[re.Pattern[str], ...]:
    """Return metadata patterns for a detected language."""
    normalized_language = language.lower().strip()
    if normalized_language == "tr":
        return _TURKISH_METADATA_PATTERNS + _SHARED_METADATA_PATTERNS
    if normalized_language == "en":
        return _ENGLISH_METADATA_PATTERNS + _SHARED_METADATA_PATTERNS
    return _TURKISH_METADATA_PATTERNS + _ENGLISH_METADATA_PATTERNS + _SHARED_METADATA_PATTERNS


def _looks_like_metadata_line(line: str, language: str) -> bool:
    """Return whether a line looks like bibliographic/article metadata."""
    stripped = line.strip()
    if not stripped:
        return False
    if _line_has_sentence_structure(stripped) and len(stripped) > 180:
        return False

    has_metadata_term = any(
        pattern.search(stripped)
        for pattern in _metadata_patterns_for_language(language)
    )
    if not has_metadata_term:
        return False

    has_field_shape = bool(re.search(r"[:/|]|\d{4}|\d+\s*[-–]\s*\d+", stripped))
    return len(stripped) <= 220 or has_field_shape


def _numeric_token_count(words: list[str]) -> int:
    """Count numeric-looking tokens in a list of words."""
    return sum(1 for word in words if _NUMBER_TOKEN_PATTERN.fullmatch(word.strip("()[];,")))


def _looks_like_table_line(line: str) -> bool:
    """Return whether a line looks like a table row or table header."""
    stripped = line.strip()
    if not stripped:
        return False

    words = stripped.split()
    word_count = len(words)
    if word_count < 3 or word_count > 48:
        return False

    numeric_tokens = _numeric_token_count(words)
    numeric_ratio = numeric_tokens / max(word_count, 1)
    decimal_or_amount_count = len(_DECIMAL_OR_AMOUNT_PATTERN.findall(stripped))
    currency_count = len(_CURRENCY_PATTERN.findall(stripped))
    tabular_spacing = bool(re.search(r"\S\s{2,}\S", stripped)) or "\t" in stripped
    slash_fragment_count = stripped.count("/")
    sentence_punctuation_count = sum(stripped.count(char) for char in ".!?")

    if numeric_tokens >= 4 and numeric_ratio >= 0.30 and sentence_punctuation_count <= 1:
        return True
    if numeric_tokens >= 3 and numeric_ratio >= 0.40 and word_count <= 16:
        return True
    if decimal_or_amount_count >= 3 and sentence_punctuation_count <= 1:
        return True
    if currency_count >= 2 and numeric_tokens >= 2:
        return True
    if tabular_spacing and numeric_tokens >= 3:
        return True
    if slash_fragment_count >= 4 and word_count <= 24:
        return True
    if (
        word_count <= 12
        and len(_TABLE_HEADER_TERM_PATTERN.findall(stripped)) >= 2
        and sentence_punctuation_count == 0
    ):
        return True

    short_column_tokens = sum(1 for word in words if len(word.strip(".,;:()")) <= 3)
    return word_count <= 16 and short_column_tokens >= 8 and sentence_punctuation_count == 0


def _looks_like_repeated_header_footer_line(line: str, count: int) -> bool:
    """Return whether a repeated line is likely a page header/footer."""
    stripped = line.strip()
    if count < 2 or not stripped:
        return False
    if _line_has_sentence_structure(stripped):
        return False

    words = stripped.split()
    if len(words) > 16 or len(stripped) > 140:
        return False

    has_header_shape = bool(re.search(r"[/|:–-]|\d", stripped))
    compact_title_like = sum(word[:1].isupper() for word in words) >= max(2, len(words) // 2)
    return has_header_shape or compact_title_like or len(words) <= 8


def _remove_embedded_repeated_fragments(
    line: str,
    repeated_fragments: list[str],
) -> str:
    """Remove repeated header/footer fragments embedded inside longer lines."""
    cleaned_line = line

    for fragment in repeated_fragments:
        stripped_fragment = fragment.strip()
        if len(stripped_fragment) < 12:
            continue

        escaped_fragment = re.escape(stripped_fragment)

        joined_fragment_pattern = re.compile(
            rf"([A-Za-zÇĞİÖŞÜçğıöşü]{{3,}}){escaped_fragment}\s+([a-zçğıöşü]{{3,}})",
            flags=re.IGNORECASE,
        )
        cleaned_line = joined_fragment_pattern.sub(
            lambda match: f"{match.group(1)}{match.group(2)}",
            cleaned_line,
        )

        spaced_fragment_pattern = re.compile(
            rf"\s+{escaped_fragment}\s+",
            flags=re.IGNORECASE,
        )
        cleaned_line = spaced_fragment_pattern.sub(" ", cleaned_line)

        attached_fragment_pattern = re.compile(
            rf"([A-Za-zÇĞİÖŞÜçğıöşü]{{3,}}){escaped_fragment}",
            flags=re.IGNORECASE,
        )
        cleaned_line = attached_fragment_pattern.sub(r"\1", cleaned_line)

    return re.sub(r"\s{2,}", " ", cleaned_line).strip()


def _is_common_section_heading(line: str) -> bool:
    """Return whether a short line is a general academic section heading."""
    normalized = re.sub(r"^\d+(?:\.\d+)*\.?\s*", "", line.strip().lower())
    normalized = normalized.strip(" .:-–—")
    return normalized in _COMMON_SECTION_HEADINGS


def remove_academic_metadata_lines(text: str, language: str) -> tuple[str, int]:
    """Remove general academic metadata lines and return removal count."""
    lines = text.splitlines()
    kept_lines: list[str] = []

    for line in lines:
        if _looks_like_metadata_line(line, language):
            continue
        kept_lines.append(line)

    return "\n".join(kept_lines), _count_removed_lines(lines, kept_lines)


def remove_table_like_lines(text: str) -> tuple[str, int]:
    """Remove table-like row/header lines and return removal count."""
    lines = text.splitlines()
    kept_lines: list[str] = []

    for line in lines:
        if _looks_like_table_line(line):
            continue
        kept_lines.append(line)

    return "\n".join(kept_lines), _count_removed_lines(lines, kept_lines)


def remove_repeated_header_footer_lines(text: str) -> tuple[str, int]:
    """Remove repeated general page header/footer lines and return removal count."""
    lines = text.splitlines()
    line_counts: dict[str, int] = {}

    for line in lines:
        normalized = _normalize_line_for_counting(line)
        if normalized:
            line_counts[normalized] = line_counts.get(normalized, 0) + 1

    repeated_fragments = [
        line.strip()
        for line in lines
        if line.strip()
        and _looks_like_repeated_header_footer_line(
            line.strip(),
            line_counts.get(_normalize_line_for_counting(line), 0),
        )
    ]

    kept_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        normalized = _normalize_line_for_counting(line)
        count = line_counts.get(normalized, 0)

        if stripped and _looks_like_repeated_header_footer_line(stripped, count):
            continue
        kept_lines.append(_remove_embedded_repeated_fragments(line, repeated_fragments))

    return "\n".join(kept_lines), _count_removed_lines(lines, kept_lines)


def remove_noisy_short_lines(text: str) -> tuple[str, int]:
    """Remove very short PDF fragments and return removal count."""
    lines = text.splitlines()
    kept_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            kept_lines.append(line)
            continue

        if len(stripped) < 3:
            continue
        if PAGE_NUMBER_PATTERN.fullmatch(stripped):
            continue
        if _is_mostly_numbers_or_symbols(stripped):
            continue
        if re.fullmatch(r"(?:[A-ZÇĞİÖŞÜ]\.?\s*){1,3}", stripped):
            continue
        if _is_common_section_heading(stripped):
            kept_lines.append(line)
            continue
        if len(stripped.split()) <= 2 and not re.search(r"[.!?]$", stripped):
            continue

        kept_lines.append(line)

    return "\n".join(kept_lines), _count_removed_lines(lines, kept_lines)


def _safe_cleaning_step(
    text: str,
    default_count: int,
    step,
    *args,
) -> tuple[str, int]:
    """Run a cleaning step without letting it crash the app."""
    try:
        return step(text, *args)
    except Exception:
        return text, default_count


def clean_for_summarization(text: str, language: str) -> tuple[str, dict[str, int]]:
    """Apply the line-preserving summarization cleaning pipeline.

    The input should be raw or lightly normalized PDF text, because repeated
    headers, metadata, tables, and short fragments must be removed before
    whitespace is collapsed.
    """
    debug = {
        "removed_repeated_line_count": 0,
        "removed_metadata_line_count": 0,
        "removed_table_line_count": 0,
        "removed_short_noise_line_count": 0,
        "repaired_hyphenation_count": 0,
    }

    result = text
    try:
        result, debug["repaired_hyphenation_count"] = _fix_pdf_hyphenation_with_count(
            result
        )
    except Exception:
        pass

    for step in (remove_urls, remove_emails, remove_doi_patterns):
        try:
            result = step(result)
        except Exception:
            pass

    try:
        result = remove_references_section(result, language)
    except Exception:
        pass

    result, count = _safe_cleaning_step(result, 0, remove_repeated_header_footer_lines)
    debug["removed_repeated_line_count"] = count

    result, count = _safe_cleaning_step(result, 0, remove_academic_metadata_lines, language)
    debug["removed_metadata_line_count"] = count

    result, count = _safe_cleaning_step(result, 0, remove_table_like_lines)
    debug["removed_table_line_count"] = count

    result, count = _safe_cleaning_step(result, 0, remove_noisy_short_lines)
    debug["removed_short_noise_line_count"] = count

    try:
        result, second_pass_count = _fix_pdf_hyphenation_with_count(result)
        debug["repaired_hyphenation_count"] += second_pass_count
    except Exception:
        pass

    try:
        result = "\n".join(line.strip() for line in result.splitlines())
        result = re.sub(r"\n{3,}", "\n\n", result)
        result = remove_extra_whitespace(result)
    except Exception:
        result = text

    return result, debug
