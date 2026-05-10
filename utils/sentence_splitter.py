"""Sentence splitting helpers for Turkish and English text."""

from __future__ import annotations

import re


MIN_SENTENCE_CHARACTERS = 20
MIN_SENTENCE_WORDS = 3
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    """Split Turkish and English text into simple sentence units.

    Args:
        text: Input text to split.

    Returns:
        A list of stripped sentences, excluding very short fragments.
    """
    if not text.strip():
        return []

    candidate_sentences = [
        sentence.strip()
        for sentence in SENTENCE_SPLIT_PATTERN.split(text)
        if sentence.strip()
    ]

    sentences = [
        sentence
        for sentence in candidate_sentences
        if len(sentence) >= MIN_SENTENCE_CHARACTERS
        and len(sentence.split()) >= MIN_SENTENCE_WORDS
    ]

    # TODO: Replace this regex splitter with a language-aware sentence segmenter.
    # TODO: Add abbreviation handling for academic citations and section labels.
    return sentences
