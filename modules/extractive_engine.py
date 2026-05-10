"""Extractive summarization engines."""

from __future__ import annotations

import math
import re
from typing import Any

from utils.sentence_splitter import split_sentences


EMAIL_PATTERN = re.compile(r"\b[\w.%+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
URL_PATTERN = re.compile(r"\b(?:https?://|www\.)\S+", flags=re.IGNORECASE)
BULLET_START_PATTERN = re.compile(r"^\s*(?:[•●▪◦*]+|[-–—]{2,})")
AFFILIATION_PATTERN = re.compile(
    r"\b(?:academy|centre|center|college|department|faculty|institute|laboratory|"
    r"lab|school|technical university|university)\b",
    flags=re.IGNORECASE,
)
ADDRESS_PATTERN = re.compile(
    r"\b(?:avenue|berardi|blvd|boulevard|city|denmark|italy|road|street|turkey|"
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


def is_valid_summary_sentence(sentence: str) -> bool:
    """Return whether a sentence is suitable for TF-IDF summarization."""
    stripped_sentence = sentence.strip()
    word_count = len(stripped_sentence.split())

    if len(stripped_sentence) < 40 or len(stripped_sentence) > 700:
        return False
    if word_count < 6:
        return False
    if BULLET_START_PATTERN.match(stripped_sentence):
        return False
    if EMAIL_PATTERN.search(stripped_sentence) or URL_PATTERN.search(stripped_sentence):
        return False

    lowered_sentence = stripped_sentence.lower()
    if "published by" in lowered_sentence or "copyright" in lowered_sentence:
        return False
    if _is_mostly_numbers(stripped_sentence):
        return False
    if _looks_like_affiliation_or_address(stripped_sentence):
        return False

    comma_count = stripped_sentence.count(",")
    if comma_count >= 4 and comma_count / max(word_count, 1) > 0.12:
        return False

    return True


def summarize_with_tfidf(text: str, summary_ratio: float = 0.25) -> dict[str, Any]:
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
    valid_sentence_items = [
        (index, sentence)
        for index, sentence in enumerate(original_sentences)
        if is_valid_summary_sentence(sentence)
    ]
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
        vectorizer = TfidfVectorizer()
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


def summarize_with_textrank(text: str, summary_ratio: float = 0.25) -> dict[str, Any]:
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
    valid_sentence_items = [
        (index, sentence)
        for index, sentence in enumerate(original_sentences)
        if is_valid_summary_sentence(sentence)
    ]
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
        vectorizer = TfidfVectorizer()
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
