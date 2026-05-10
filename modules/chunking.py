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
