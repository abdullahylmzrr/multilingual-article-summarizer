"""Text preprocessing helpers for future summarization pipelines."""

from __future__ import annotations

from utils.text_cleaner import normalize_whitespace


def preprocess_text(text: str, lowercase: bool = False) -> str:
    """Apply basic preprocessing to raw text.

    Args:
        text: Raw input text.
        lowercase: Whether to lowercase the normalized text.

    Returns:
        Preprocessed text.
    """
    cleaned_text = normalize_whitespace(text)
    if lowercase:
        return cleaned_text.lower()

    return cleaned_text
