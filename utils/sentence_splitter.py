"""Sentence splitting helpers."""

from __future__ import annotations

import re


def split_sentences(text: str) -> list[str]:
    """Split text into sentences using a simple punctuation-based rule."""
    if not text.strip():
        return []

    return [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text) if sentence.strip()]
