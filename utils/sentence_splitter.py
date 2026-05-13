"""Sentence splitting helpers for Turkish and English text."""

from __future__ import annotations

import re


MIN_SENTENCE_CHARACTERS = 20
MIN_SENTENCE_WORDS = 3
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")
DECIMAL_DOT_PLACEHOLDER = "<DECIMAL_DOT>"
ABBREVIATION_DOT_PLACEHOLDER = "<ABBR_DOT>"
COMMON_ABBREVIATIONS = (
    "Dr.",
    "Prof.",
    "Doç.",
    "Yrd.",
    "Arş.",
    "Mr.",
    "Mrs.",
    "Ms.",
    "Vol.",
    "No.",
    "Fig.",
    "Şek.",
    "Bkz.",
    "vb.",
    "vs.",
    "örn.",
    "e.g.",
    "i.e.",
)


def _protect_sentence_boundaries(text: str) -> str:
    """Protect decimals and common abbreviations before regex splitting."""
    protected_text = re.sub(
        r"(?<=\d)\.(?=\d)",
        DECIMAL_DOT_PLACEHOLDER,
        text,
    )

    for abbreviation in COMMON_ABBREVIATIONS:
        protected_abbreviation = abbreviation.replace(".", ABBREVIATION_DOT_PLACEHOLDER)
        protected_text = protected_text.replace(abbreviation, protected_abbreviation)

    return protected_text


def _restore_sentence_boundaries(text: str) -> str:
    """Restore protected dots after sentence splitting."""
    return (
        text.replace(DECIMAL_DOT_PLACEHOLDER, ".")
        .replace(ABBREVIATION_DOT_PLACEHOLDER, ".")
    )


def split_sentences(text: str) -> list[str]:
    """Split Turkish and English text into simple sentence units.

    Args:
        text: Input text to split.

    Returns:
        A list of stripped sentences, excluding very short fragments.
    """
    if not text.strip():
        return []

    protected_text = _protect_sentence_boundaries(text)
    candidate_sentences = [
        _restore_sentence_boundaries(sentence.strip())
        for sentence in SENTENCE_SPLIT_PATTERN.split(protected_text)
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
