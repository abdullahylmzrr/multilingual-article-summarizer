"""Extractive summarization engine placeholders."""

from __future__ import annotations


def summarize_with_tfidf(text: str, sentence_count: int = 5) -> str:
    """Placeholder for future TF-IDF extractive summarization."""
    # TODO: Implement TF-IDF based sentence scoring and summary generation.
    raise NotImplementedError("TF-IDF summarization is not implemented yet.")


def summarize_with_textrank(text: str, sentence_count: int = 5) -> str:
    """Placeholder for future TextRank extractive summarization."""
    # TODO: Implement TextRank graph-based extractive summarization.
    raise NotImplementedError("TextRank summarization is not implemented yet.")
