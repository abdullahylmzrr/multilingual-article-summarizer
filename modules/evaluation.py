"""Evaluation utilities for comparing summarization methods."""

from __future__ import annotations


def count_words(text: str) -> int:
    """Count whitespace-separated words in text."""
    return len(text.split())


def count_sentences(sentences: list[str]) -> int:
    """Count sentences in a sentence list."""
    return len(sentences)


def calculate_compression_ratio(
    original_word_count: int,
    summary_word_count: int,
) -> float:
    """Calculate summary length as a fraction of original length."""
    if original_word_count <= 0:
        return 0.0

    return summary_word_count / original_word_count


def _normalize_sentence(sentence: str) -> str:
    """Normalize a sentence for exact-overlap comparison."""
    return " ".join(sentence.lower().strip().split())


def calculate_sentence_overlap(
    sentences_a: list[str],
    sentences_b: list[str],
) -> dict[str, object]:
    """Calculate overlap metrics between two selected sentence lists."""
    normalized_a = {_normalize_sentence(sentence): sentence for sentence in sentences_a}
    normalized_b = {_normalize_sentence(sentence): sentence for sentence in sentences_b}

    sentences_a_set = set(normalized_a)
    sentences_b_set = set(normalized_b)
    common_sentence_keys = sentences_a_set.intersection(sentences_b_set)
    unique_sentence_keys = sentences_a_set.union(sentences_b_set)

    common_sentence_count = len(common_sentence_keys)
    total_unique_sentence_count = len(unique_sentence_keys)
    jaccard_similarity = (
        common_sentence_count / total_unique_sentence_count
        if total_unique_sentence_count
        else 0.0
    )
    overlap_denominator = min(len(sentences_a_set), len(sentences_b_set))
    overlap_percentage = (
        common_sentence_count / overlap_denominator * 100
        if overlap_denominator
        else 0.0
    )
    common_sentences = [
        sentence
        for sentence in sentences_a
        if _normalize_sentence(sentence) in common_sentence_keys
    ]

    return {
        "common_sentence_count": common_sentence_count,
        "total_unique_sentence_count": total_unique_sentence_count,
        "jaccard_similarity": jaccard_similarity,
        "overlap_percentage": overlap_percentage,
        "common_sentences": common_sentences,
    }


def compare_summaries(
    tfidf_result: dict,
    textrank_result: dict,
    original_text: str,
) -> dict[str, object]:
    """Compare TF-IDF and TextRank summary outputs with lightweight metrics."""
    original_word_count = count_words(original_text)
    tfidf_word_count = count_words(str(tfidf_result.get("summary", "")))
    textrank_word_count = count_words(str(textrank_result.get("summary", "")))

    return {
        "original_word_count": original_word_count,
        "tfidf_word_count": tfidf_word_count,
        "textrank_word_count": textrank_word_count,
        "tfidf_compression_ratio": calculate_compression_ratio(
            original_word_count,
            tfidf_word_count,
        ),
        "textrank_compression_ratio": calculate_compression_ratio(
            original_word_count,
            textrank_word_count,
        ),
        "sentence_overlap": calculate_sentence_overlap(
            list(tfidf_result.get("selected_sentences", [])),
            list(textrank_result.get("selected_sentences", [])),
        ),
    }
