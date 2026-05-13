"""Extractive summarization engines."""

from __future__ import annotations

import math
import re
from typing import Any

from utils.sentence_splitter import split_sentences
from utils.stopwords import get_stopwords


EMAIL_PATTERN = re.compile(r"\b[\w.%+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
URL_PATTERN = re.compile(r"\b(?:https?://|www\.)\S+", flags=re.IGNORECASE)
BULLET_START_PATTERN = re.compile(r"^\s*(?:[•●▪◦*]+|[-–—]{2,})")
AFFILIATION_PATTERN = re.compile(
    r"\b(?:academy|centre|center|college|department|faculty|institute|laboratory|"
    r"lab|school|technical university|university)\b",
    flags=re.IGNORECASE,
)
ADDRESS_PATTERN = re.compile(
    r"\b(?:avenue|blvd|boulevard|city|denmark|italy|road|street|turkey|"
    r"united kingdom|united states|via)\b",
    flags=re.IGNORECASE,
)


def _fallback_tfidf_response(
    summary: str,
    selected_sentences: list[str],
    original_sentence_count: int,
    valid_sentence_count: int,
    summary_ratio: float,
    message: str,
) -> dict[str, Any]:
    """Build a consistent TF-IDF fallback response."""
    stripped_summary = summary.strip()

    return {
        "method": "TF-IDF",
        "summary": stripped_summary,
        "selected_sentences": selected_sentences,
        "sentence_scores": [],
        "sentence_count": original_sentence_count,
        "original_sentence_count": original_sentence_count,
        "valid_sentence_count": valid_sentence_count,
        "selected_sentence_count": len(selected_sentences),
        "summary_ratio": summary_ratio,
        "message": message,
    }


def _normalize_summary_ratio(summary_ratio: float) -> float:
    """Keep the summary ratio inside a useful range."""
    if summary_ratio <= 0:
        return 0.25
    if summary_ratio > 1:
        return 1.0
    return summary_ratio


def _is_mostly_numbers(sentence: str) -> bool:
    """Return whether a sentence has too little alphabetic content."""
    alphabetic_count = sum(character.isalpha() for character in sentence)
    numeric_count = sum(character.isdigit() for character in sentence)

    if numeric_count == 0:
        return False

    return alphabetic_count / max(len(sentence), 1) < 0.35 or numeric_count > alphabetic_count


def _looks_like_affiliation_or_address(sentence: str) -> bool:
    """Return whether a sentence looks like an author affiliation/address line."""
    comma_count = sentence.count(",")
    digit_count = sum(character.isdigit() for character in sentence)
    starts_with_index = bool(re.match(r"^\s*\d+\s+", sentence))
    has_affiliation = bool(AFFILIATION_PATTERN.search(sentence))
    has_address = bool(ADDRESS_PATTERN.search(sentence))

    if starts_with_index and has_affiliation:
        return True
    if has_affiliation and has_address and comma_count >= 1:
        return True
    if has_affiliation and comma_count >= 2 and digit_count >= 1:
        return True

    return False


DOI_SENTENCE_PATTERN = re.compile(
    r"\b(?:doi\s*:|10\.\d{4,9}/)", flags=re.IGNORECASE
)
METADATA_KEYWORD_PATTERN = re.compile(
    r"\b(?:issn|e-issn|p-issn|doi|cilt|sayı|sayfa|volume|vol\.?|issue|"
    r"pages?|received|accepted|geliş\s+tarihi|kabul\s+tarihi|copyright|"
    r"published\s+by)\b",
    flags=re.IGNORECASE,
)
CURRENCY_SENTENCE_PATTERN = re.compile(r"[€$]|\bTL\b|\bEuro\b", flags=re.IGNORECASE)
DECIMAL_OR_AMOUNT_PATTERN = re.compile(
    r"\d+[.,]\d+|\d+\s*(?:%|€|\$|£|₺|\bTL\b|\bEuro\b)",
    flags=re.IGNORECASE,
)
TABLE_HEADER_TERM_PATTERN = re.compile(
    r"\b(?:adet|tutar\w*|toplam|oran|yüzde|type|count|amount|total|rate|percent)\b",
    flags=re.IGNORECASE,
)
COLUMN_HEADER_PAIR_PATTERN = re.compile(
    r"\b(?:proje|project)\s+tutar\w*.*\b(?:hibe|grant)\s+tutar\w*|"
    r"\b(?:hibe|grant)\s+tutar\w*.*\b(?:proje|project)\s+tutar\w*",
    flags=re.IGNORECASE,
)
SHORT_HEADER_BEFORE_BODY_PATTERN = re.compile(
    r"\b[A-ZÇĞİÖŞÜ][A-Za-zÇĞİÖŞÜçğıöşü.'-]{2,}\s+"
    r"(?:Üni\.?|Univ\.?|Dergisi|Journal)\.?(?:\s+[a-zçğıöşü]{3,}|\s*$)",
)
TOKEN_PATTERN = r"(?u)\b[A-Za-zÇĞİÖŞÜçğıöşü]{2,}\b"
LEADING_SECTION_TERMS = {
    "abstract",
    "discussion",
    "giriş",
    "introduction",
    "method",
    "methodology",
    "results",
    "sonuç",
    "sonuçlar",
    "summary",
    "tartışma",
    "yöntem",
    "özet",
}
ENGLISH_FUNCTION_WORDS = {
    "a",
    "also",
    "and",
    "are",
    "as",
    "been",
    "for",
    "from",
    "have",
    "in",
    "is",
    "of",
    "on",
    "that",
    "the",
    "their",
    "this",
    "to",
    "with",
}
ENGLISH_ABSTRACT_MARKER_PATTERN = re.compile(
    r"\b(?:from\s+this\s+date|principle\s+of\s+the|sustainable\s+development|"
    r"determination\s+of|constitution\s+of|have\s+been\s+realized)\b",
    flags=re.IGNORECASE,
)
LEADING_SECTION_HEADING_PATTERN = re.compile(
    r"^(?:\d+(?:\.\d+)*\.?\s*)?"
    r"(?:TARTIŞMA\s+VE\s+SONUÇ|TARTIŞMA|SONUÇLAR?|GİRİŞ|ÖZET|YÖNTEM|"
    r"DISCUSSION\s+AND\s+CONCLUSION|DISCUSSION|CONCLUSIONS?|INTRODUCTION|"
    r"ABSTRACT|RESULTS|METHOD(?:OLOGY)?|SUMMARY)"
    r"\s+(?P<body>.{40,})$",
    flags=re.IGNORECASE,
)


def _vectorizer_kwargs(
    language: str,
    ngram_range: tuple[int, int],
) -> dict[str, Any]:
    """Build conservative TF-IDF vectorizer settings for academic text."""
    stopwords = sorted(get_stopwords(language))
    return {
        "lowercase": True,
        "max_df": 0.90,
        "min_df": 1,
        "ngram_range": ngram_range,
        "stop_words": stopwords or None,
        "token_pattern": TOKEN_PATTERN,
    }


def _strip_leading_section_heading(sentence: str) -> str:
    """Remove a leading academic section heading before a body sentence."""
    stripped_sentence = re.sub(r"\s+", " ", sentence).strip()
    heading_match = LEADING_SECTION_HEADING_PATTERN.match(stripped_sentence)
    if heading_match:
        return heading_match.group("body").strip()

    match = re.match(
        r"^(?P<heading>[A-ZÇĞİÖŞÜ][A-Za-zÇĞİÖŞÜçğıöşü&/-]*"
        r"(?:\s+[A-ZÇĞİÖŞÜVEAND&/-][A-Za-zÇĞİÖŞÜçğıöşü&/-]*){0,5})"
        r"\s+(?P<body>[A-ZÇĞİÖŞÜ0-9][^.!?]{40,})$",
        stripped_sentence,
    )
    if not match:
        return stripped_sentence

    heading = match.group("heading")
    normalized_heading = re.sub(r"[^A-Za-zÇĞİÖŞÜçğıöşü]+", " ", heading).lower()
    heading_terms = set(normalized_heading.split())
    if heading_terms & LEADING_SECTION_TERMS:
        return match.group("body").strip()

    return stripped_sentence


def _normalize_candidate_sentence(sentence: str) -> str:
    """Normalize a split sentence before extractive validation and scoring."""
    normalized_sentence = re.sub(r"\s+", " ", sentence).strip()
    return _strip_leading_section_heading(normalized_sentence)


def _looks_like_english_abstract_sentence(sentence: str, language: str) -> bool:
    """Detect English abstract sentences inside Turkish academic PDFs."""
    if language.lower().strip() != "tr":
        return False
    if ENGLISH_ABSTRACT_MARKER_PATTERN.search(sentence):
        return True

    tokens = re.findall(r"[A-Za-z]+", sentence.lower())
    if len(tokens) < 8:
        return False

    function_word_count = sum(token in ENGLISH_FUNCTION_WORDS for token in tokens)
    has_turkish_characters = bool(re.search(r"[çğıöşüÇĞİÖŞÜ]", sentence))
    return not has_turkish_characters and function_word_count / len(tokens) >= 0.20


def _starts_with_broken_lowercase(sentence: str) -> bool:
    """Return whether a candidate starts like a broken sentence fragment."""
    stripped_sentence = sentence.lstrip(" \t\n\r\"'“”‘’([{")
    return bool(stripped_sentence) and stripped_sentence[0].islower()


def _is_valid_candidate_sentence(sentence: str, language: str) -> bool:
    """Apply language-aware extractive candidate checks before scoring."""
    if _starts_with_broken_lowercase(sentence):
        return False
    if _looks_like_english_abstract_sentence(sentence, language):
        return False

    return is_valid_summary_sentence(sentence)


def _build_valid_sentence_items(
    original_sentences: list[str],
    language: str,
) -> list[tuple[int, str]]:
    """Normalize split sentences and keep valid extractive candidates."""
    valid_sentence_items = []

    for index, sentence in enumerate(original_sentences):
        normalized_candidate = _normalize_candidate_sentence(sentence)
        if _is_valid_candidate_sentence(normalized_candidate, language):
            valid_sentence_items.append((index, normalized_candidate))

    return valid_sentence_items


def _numeric_token_count(words: list[str]) -> int:
    """Count numeric-looking tokens in a sentence."""
    return sum(
        1
        for word in words
        if re.fullmatch(r"[\d.,:/-]+%?", word.strip("()[];,:"))
    )


def _looks_like_table_sentence(sentence: str) -> bool:
    """Return whether a sentence looks like a table row rather than prose."""
    words = sentence.split()
    word_count = len(words)
    numeric_tokens = _numeric_token_count(words)
    numeric_ratio = numeric_tokens / max(word_count, 1)
    decimal_or_amount_count = len(DECIMAL_OR_AMOUNT_PATTERN.findall(sentence))
    currency_count = len(CURRENCY_SENTENCE_PATTERN.findall(sentence))
    table_header_count = len(TABLE_HEADER_TERM_PATTERN.findall(sentence))
    slash_count = sentence.count("/")
    sentence_punctuation_count = sum(sentence.count(char) for char in ".!?")

    if COLUMN_HEADER_PAIR_PATTERN.search(sentence):
        return True
    if sentence.count("(Euro)") >= 2 or sentence.count("(EUR)") >= 2:
        return True
    if numeric_tokens >= 4 and numeric_ratio > 0.30 and sentence_punctuation_count <= 1:
        return True
    if numeric_tokens >= 3 and numeric_ratio >= 0.40 and word_count <= 16:
        return True
    if decimal_or_amount_count >= 3 and sentence_punctuation_count <= 1:
        return True
    if currency_count >= 2 and numeric_tokens >= 2:
        return True
    if currency_count >= 2 and table_header_count >= 2:
        return True
    if table_header_count >= 3 and word_count <= 60 and sentence_punctuation_count <= 2:
        return True
    if slash_count >= 4 and word_count <= 28:
        return True
    if (
        word_count <= 12
        and table_header_count >= 2
        and sentence_punctuation_count == 0
    ):
        return True

    short_tokens = sum(1 for word in words if len(word.strip(".,;:()")) <= 3)
    return word_count <= 18 and short_tokens >= 9 and sentence_punctuation_count == 0


def _has_too_many_uppercase_abbreviations(sentence: str) -> bool:
    """Return whether a sentence is dominated by uppercase abbreviations."""
    words = sentence.split()
    if len(words) < 8:
        return False

    uppercase_tokens = [
        word
        for word in words
        if re.fullmatch(
            r"[A-ZÇĞİÖŞÜ]{2,}(?:[-/][A-ZÇĞİÖŞÜ]{2,})?",
            word.strip(".,;:()"),
        )
    ]
    return len(uppercase_tokens) >= 5 and len(uppercase_tokens) / len(words) > 0.35


def _has_obvious_broken_pdf_fragment(sentence: str) -> bool:
    """Return whether a sentence contains structural PDF join artifacts."""
    if re.search(r"[a-zçğıöşü]{3,}[A-ZÇĞİÖŞÜ][A-Za-zÇĞİÖŞÜçğıöşü]{3,}", sentence):
        return True
    if re.search(
        r"\b[A-ZÇĞİÖŞÜ]?[a-zçğıöşü]{8,}\s+"
        r"(?:(?:ve|and|of|the|in|de|da|ile)\s+|"
        r"[A-ZÇĞİÖŞÜ][A-Za-zÇĞİÖŞÜçğıöşü’'-]{2,}\s+){4,}"
        r"[a-zçğıöşü]{3,}\b",
        sentence,
    ):
        return True

    alphabetic_tokens = re.findall(r"[A-Za-zÇĞİÖŞÜçğıöşü]+", sentence)
    return any(len(token) > 34 for token in alphabetic_tokens)


def _has_short_embedded_header_fragment(sentence: str) -> bool:
    """Return whether a short header fragment interrupts a body sentence."""
    if SHORT_HEADER_BEFORE_BODY_PATTERN.search(sentence):
        return True

    return bool(
        re.search(
            r"\b(?:cilt|volume|sayı|issue)\s*[/|:-]\s*(?:cilt|volume|sayı|issue)\b",
            sentence,
            flags=re.IGNORECASE,
        )
    )


def is_valid_summary_sentence(sentence: str) -> bool:
    """Return whether a sentence is suitable for extractive summarization.

    Rejects metadata fragments, table rows, and overly noisy text while
    keeping valid academic content (dates, citations, Turkish characters).
    """
    stripped_sentence = sentence.strip()
    word_count = len(stripped_sentence.split())

    if len(stripped_sentence) < 40 or len(stripped_sentence) > 900:
        return False
    if word_count < 6:
        return False
    if BULLET_START_PATTERN.match(stripped_sentence):
        return False
    if EMAIL_PATTERN.search(stripped_sentence) or URL_PATTERN.search(stripped_sentence):
        return False
    if DOI_SENTENCE_PATTERN.search(stripped_sentence):
        return False

    if _is_mostly_numbers(stripped_sentence):
        return False
    if _looks_like_affiliation_or_address(stripped_sentence):
        return False

    if METADATA_KEYWORD_PATTERN.search(stripped_sentence) and (
        word_count < 26 or re.search(r"[:/|]\s*\d|\d{4}", stripped_sentence)
    ):
        return False
    if _looks_like_table_sentence(stripped_sentence):
        return False
    if _has_too_many_uppercase_abbreviations(stripped_sentence):
        return False
    if _has_obvious_broken_pdf_fragment(stripped_sentence):
        return False
    if _has_short_embedded_header_fragment(stripped_sentence):
        return False
    if re.search(r"(?:<[^>]+>|/>|&nbsp;)", stripped_sentence, flags=re.IGNORECASE):
        return False

    comma_count = stripped_sentence.count(",")
    if comma_count >= 4 and comma_count / max(word_count, 1) > 0.12:
        return False

    return not ("/" in stripped_sentence and stripped_sentence.count("/") >= 4 and word_count < 30)


def summarize_with_tfidf(
    text: str,
    summary_ratio: float = 0.25,
    language: str = "unknown",
    ngram_range: tuple[int, int] = (1, 1),
) -> dict[str, Any]:
    """Generate an extractive summary with sentence-level TF-IDF scores.

    Args:
        text: Cleaned readable text to summarize.
        summary_ratio: Fraction of sentences to include in the summary.

    Returns:
        Dictionary containing the summary, selected sentences, scores, and metadata.
    """
    normalized_text = text.strip()
    normalized_ratio = _normalize_summary_ratio(summary_ratio)

    if not normalized_text:
        return _fallback_tfidf_response(
            summary="",
            selected_sentences=[],
            original_sentence_count=0,
            valid_sentence_count=0,
            summary_ratio=normalized_ratio,
            message="Input text is empty.",
        )

    original_sentences = split_sentences(normalized_text)
    original_sentence_count = len(original_sentences)
    valid_sentence_items = _build_valid_sentence_items(original_sentences, language)
    valid_sentences = [sentence for _, sentence in valid_sentence_items]
    valid_sentence_count = len(valid_sentences)

    if original_sentence_count == 0:
        return _fallback_tfidf_response(
            summary=normalized_text,
            selected_sentences=[],
            original_sentence_count=0,
            valid_sentence_count=0,
            summary_ratio=normalized_ratio,
            message="No valid sentences were found after sentence splitting.",
        )

    if valid_sentence_count == 0:
        return _fallback_tfidf_response(
            summary=normalized_text,
            selected_sentences=[],
            original_sentence_count=original_sentence_count,
            valid_sentence_count=0,
            summary_ratio=normalized_ratio,
            message="No sentences passed the TF-IDF quality filter.",
        )

    if valid_sentence_count <= 2:
        return {
            "method": "TF-IDF",
            "summary": " ".join(valid_sentences),
            "selected_sentences": valid_sentences,
            "sentence_scores": [
                {"index": original_index, "sentence": sentence, "score": 0.0}
                for original_index, sentence in valid_sentence_items
            ],
            "sentence_count": original_sentence_count,
            "original_sentence_count": original_sentence_count,
            "valid_sentence_count": valid_sentence_count,
            "selected_sentence_count": valid_sentence_count,
            "summary_ratio": normalized_ratio,
            "message": "Too few valid sentences for TF-IDF ranking; returned valid sentences.",
        }

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError:
        return _fallback_tfidf_response(
            summary=normalized_text,
            selected_sentences=valid_sentences,
            original_sentence_count=original_sentence_count,
            valid_sentence_count=valid_sentence_count,
            summary_ratio=normalized_ratio,
            message="scikit-learn is required for TF-IDF summarization.",
        )

    try:
        vectorizer = TfidfVectorizer(**_vectorizer_kwargs(language, ngram_range))
        tfidf_matrix = vectorizer.fit_transform(valid_sentences)
    except ValueError as error:
        if "empty vocabulary" in str(error).lower():
            return _fallback_tfidf_response(
                summary=normalized_text,
                selected_sentences=[],
                original_sentence_count=original_sentence_count,
                valid_sentence_count=valid_sentence_count,
                summary_ratio=normalized_ratio,
                message="TF-IDF could not build a vocabulary from this text.",
            )
        raise

    raw_scores = tfidf_matrix.sum(axis=1).A1
    sentence_scores = [
        {
            "index": valid_sentence_items[index][0],
            "sentence": sentence,
            "score": float(raw_scores[index]),
        }
        for index, sentence in enumerate(valid_sentences)
    ]

    selected_count = max(1, math.ceil(valid_sentence_count * normalized_ratio))
    selected_count = min(selected_count, valid_sentence_count)
    top_indices = sorted(
        range(valid_sentence_count),
        key=lambda index: (raw_scores[index], -index),
        reverse=True,
    )[:selected_count]
    selected_indices = sorted(top_indices)
    selected_sentences = [valid_sentences[index] for index in selected_indices]

    return {
        "method": "TF-IDF",
        "summary": " ".join(selected_sentences),
        "selected_sentences": selected_sentences,
        "sentence_scores": sentence_scores,
        "sentence_count": original_sentence_count,
        "original_sentence_count": original_sentence_count,
        "valid_sentence_count": valid_sentence_count,
        "selected_sentence_count": selected_count,
        "summary_ratio": normalized_ratio,
    }


def _fallback_textrank_response(
    summary: str,
    selected_sentences: list[str],
    original_sentence_count: int,
    valid_sentence_count: int,
    summary_ratio: float,
    message: str,
) -> dict[str, Any]:
    """Build a consistent TextRank fallback response."""
    return {
        "method": "TextRank",
        "summary": summary.strip(),
        "selected_sentences": selected_sentences,
        "sentence_scores": [],
        "sentence_count": original_sentence_count,
        "original_sentence_count": original_sentence_count,
        "valid_sentence_count": valid_sentence_count,
        "selected_sentence_count": len(selected_sentences),
        "summary_ratio": summary_ratio,
        "message": message,
    }


def summarize_with_textrank(
    text: str,
    summary_ratio: float = 0.25,
    language: str = "unknown",
    ngram_range: tuple[int, int] = (1, 1),
) -> dict[str, Any]:
    """Generate an extractive summary with TextRank sentence ranking.

    Args:
        text: Cleaned readable text to summarize.
        summary_ratio: Fraction of valid sentences to include in the summary.

    Returns:
        Dictionary containing the summary, selected sentences, scores, and metadata.
    """
    normalized_text = text.strip()
    normalized_ratio = _normalize_summary_ratio(summary_ratio)

    if not normalized_text:
        return _fallback_textrank_response(
            summary="",
            selected_sentences=[],
            original_sentence_count=0,
            valid_sentence_count=0,
            summary_ratio=normalized_ratio,
            message="Input text is empty.",
        )

    original_sentences = split_sentences(normalized_text)
    original_sentence_count = len(original_sentences)
    valid_sentence_items = _build_valid_sentence_items(original_sentences, language)
    valid_sentences = [sentence for _, sentence in valid_sentence_items]
    valid_sentence_count = len(valid_sentences)

    if original_sentence_count == 0:
        return _fallback_textrank_response(
            summary=normalized_text,
            selected_sentences=[],
            original_sentence_count=0,
            valid_sentence_count=0,
            summary_ratio=normalized_ratio,
            message="No valid sentences were found after sentence splitting.",
        )

    if valid_sentence_count == 0:
        return _fallback_textrank_response(
            summary=normalized_text,
            selected_sentences=[],
            original_sentence_count=original_sentence_count,
            valid_sentence_count=0,
            summary_ratio=normalized_ratio,
            message="No sentences passed the TextRank quality filter.",
        )

    if valid_sentence_count <= 2:
        return {
            "method": "TextRank",
            "summary": " ".join(valid_sentences),
            "selected_sentences": valid_sentences,
            "sentence_scores": [
                {"index": original_index, "sentence": sentence, "score": 0.0}
                for original_index, sentence in valid_sentence_items
            ],
            "sentence_count": original_sentence_count,
            "original_sentence_count": original_sentence_count,
            "valid_sentence_count": valid_sentence_count,
            "selected_sentence_count": valid_sentence_count,
            "summary_ratio": normalized_ratio,
            "message": "Too few valid sentences for TextRank ranking; returned valid sentences.",
        }

    try:
        import networkx as nx
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        return _fallback_textrank_response(
            summary=normalized_text,
            selected_sentences=valid_sentences,
            original_sentence_count=original_sentence_count,
            valid_sentence_count=valid_sentence_count,
            summary_ratio=normalized_ratio,
            message="networkx and scikit-learn are required for TextRank summarization.",
        )

    try:
        vectorizer = TfidfVectorizer(**_vectorizer_kwargs(language, ngram_range))
        tfidf_matrix = vectorizer.fit_transform(valid_sentences)
        similarity_matrix = cosine_similarity(tfidf_matrix)
    except ValueError as error:
        if "empty vocabulary" in str(error).lower():
            return _fallback_textrank_response(
                summary=normalized_text,
                selected_sentences=[],
                original_sentence_count=original_sentence_count,
                valid_sentence_count=valid_sentence_count,
                summary_ratio=normalized_ratio,
                message="TextRank could not build a vocabulary from this text.",
            )
        return _fallback_textrank_response(
            summary=normalized_text,
            selected_sentences=[],
            original_sentence_count=original_sentence_count,
            valid_sentence_count=valid_sentence_count,
            summary_ratio=normalized_ratio,
            message=f"TextRank vectorization failed: {error}",
        )
    except Exception as error:
        return _fallback_textrank_response(
            summary=normalized_text,
            selected_sentences=[],
            original_sentence_count=original_sentence_count,
            valid_sentence_count=valid_sentence_count,
            summary_ratio=normalized_ratio,
            message=f"TextRank cosine similarity failed: {error}",
        )

    graph = nx.Graph()
    graph.add_nodes_from(range(valid_sentence_count))

    for source_index in range(valid_sentence_count):
        for target_index in range(source_index + 1, valid_sentence_count):
            similarity_score = float(similarity_matrix[source_index][target_index])
            if similarity_score > 0:
                graph.add_edge(
                    source_index,
                    target_index,
                    weight=similarity_score,
                )

    try:
        rank_scores = nx.pagerank(graph, weight="weight")
    except Exception as error:
        return _fallback_textrank_response(
            summary=normalized_text,
            selected_sentences=[],
            original_sentence_count=original_sentence_count,
            valid_sentence_count=valid_sentence_count,
            summary_ratio=normalized_ratio,
            message=f"TextRank PageRank failed: {error}",
        )

    selected_count = max(1, math.ceil(valid_sentence_count * normalized_ratio))
    selected_count = min(selected_count, valid_sentence_count)
    top_indices = sorted(
        range(valid_sentence_count),
        key=lambda index: (rank_scores.get(index, 0.0), -index),
        reverse=True,
    )[:selected_count]
    selected_indices = sorted(top_indices)
    selected_sentences = [valid_sentences[index] for index in selected_indices]
    sentence_scores = [
        {
            "index": valid_sentence_items[index][0],
            "sentence": sentence,
            "score": float(rank_scores.get(index, 0.0)),
        }
        for index, sentence in enumerate(valid_sentences)
    ]

    return {
        "method": "TextRank",
        "summary": " ".join(selected_sentences),
        "selected_sentences": selected_sentences,
        "sentence_scores": sentence_scores,
        "sentence_count": original_sentence_count,
        "original_sentence_count": original_sentence_count,
        "valid_sentence_count": valid_sentence_count,
        "selected_sentence_count": selected_count,
        "summary_ratio": normalized_ratio,
    }
