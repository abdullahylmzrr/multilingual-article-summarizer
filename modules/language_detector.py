"""Language detection utilities for Turkish and English text."""

from __future__ import annotations

from langdetect import DetectorFactory, LangDetectException, detect

from config.settings import SUPPORTED_LANGUAGES


DetectorFactory.seed = 42


def detect_language(text: str) -> str:
    """Detect whether text is Turkish, English, or unknown.

    Args:
        text: Input document text.

    Returns:
        `"tr"` for Turkish, `"en"` for English, or `"unknown"` for unsupported
        languages and low-confidence/invalid inputs.
    """
    normalized_text = text.strip()
    if len(normalized_text) < 20:
        return "unknown"

    try:
        language = detect(normalized_text)
    except LangDetectException:
        return "unknown"

    if language in SUPPORTED_LANGUAGES:
        return language

    return "unknown"
