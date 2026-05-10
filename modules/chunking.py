"""Text chunking utilities for long academic articles."""

from __future__ import annotations


def chunk_text(text: str, max_characters: int = 4000) -> list[str]:
    """Split text into simple character-bounded chunks.

    Args:
        text: Input text to chunk.
        max_characters: Maximum number of characters per chunk.

    Returns:
        List of text chunks.
    """
    if max_characters <= 0:
        raise ValueError("max_characters must be greater than zero.")

    return [text[index : index + max_characters] for index in range(0, len(text), max_characters)]


def chunk_text_by_words(
    text: str,
    max_words: int = 450,
    overlap_words: int = 50,
) -> list[str]:
    """Split text into overlapping word-based chunks.

    Args:
        text: Input text to chunk.
        max_words: Maximum number of words per chunk.
        overlap_words: Number of words repeated from the previous chunk.

    Returns:
        List of word-bounded chunk strings.
    """
    words = text.split()
    if not words:
        return []

    if max_words <= 0:
        return [" ".join(words)]

    safe_overlap = max(0, min(overlap_words, max_words - 1))
    step_size = max_words - safe_overlap
    chunks = []

    for start_index in range(0, len(words), step_size):
        chunk_words = words[start_index : start_index + max_words]
        if not chunk_words:
            break

        chunks.append(" ".join(chunk_words))

        if start_index + max_words >= len(words):
            break

    return chunks
