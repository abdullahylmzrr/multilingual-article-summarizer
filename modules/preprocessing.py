"""Multilingual preprocessing helpers for future summarization pipelines."""

from __future__ import annotations

import importlib
import re

from utils.stopwords import get_stopwords
import utils.text_cleaner as text_cleaner


TOKEN_CLEANUP_PATTERN = re.compile(r"[^\w\sçğıöşüÇĞİÖŞÜ]", flags=re.UNICODE)
REQUIRED_TEXT_CLEANER_FUNCTIONS = {
    "remove_doi_patterns",
    "remove_emails",
    "remove_extra_whitespace",
    "remove_page_numbers",
    "remove_references_section",
    "remove_urls",
}


def _get_text_cleaner_module():
    """Return a current text_cleaner module, reloading if Streamlit cached an old one."""
    missing_functions = [
        name for name in REQUIRED_TEXT_CLEANER_FUNCTIONS if not hasattr(text_cleaner, name)
    ]

    if missing_functions:
        return importlib.reload(text_cleaner)

    return text_cleaner


def _count_words(text: str) -> int:
    """Count whitespace-separated words in text."""
    return len(text.split())


def _build_display_text(raw_text: str, language: str) -> str:
    """Create readable cleaned text for UI display."""
    cleaner = _get_text_cleaner_module()
    cleaned_text = cleaner.remove_references_section(raw_text, language)
    cleaned_text = cleaner.remove_emails(cleaned_text)
    cleaned_text = cleaner.remove_doi_patterns(cleaned_text)
    cleaned_text = cleaner.remove_urls(cleaned_text)
    cleaned_text = cleaner.remove_page_numbers(cleaned_text)
    return cleaner.remove_extra_whitespace(cleaned_text)


def _build_nlp_text(display_text: str, language: str) -> str:
    """Create lowercased stopword-filtered text for future NLP methods."""
    lowercase_text = display_text.lower()
    token_ready_text = TOKEN_CLEANUP_PATTERN.sub(" ", lowercase_text)
    stopwords = get_stopwords(language)
    tokens = [
        token
        for token in token_ready_text.split()
        if token not in stopwords and not token.isdigit()
    ]

    # TODO: Add Turkish lemmatization with Zeyrek before TF-IDF/TextRank experiments.
    # TODO: Add English lemmatization before TF-IDF/TextRank experiments.
    return " ".join(tokens)


def preprocess_text(raw_text: str, language: str) -> dict[str, object]:
    """Preprocess raw article text for display and future NLP pipelines.

    Args:
        raw_text: Raw text extracted from a PDF.
        language: Detected language code, usually ``"tr"`` or ``"en"``.

    Returns:
        Dictionary containing readable display text, NLP-ready text, and counts.
    """
    display_text = _build_display_text(raw_text, language)
    nlp_text = _build_nlp_text(display_text, language)

    return {
        "display_text": display_text,
        "nlp_text": nlp_text,
        "stats": {
            "raw_character_count": len(raw_text),
            "cleaned_character_count": len(display_text),
            "raw_word_count": _count_words(raw_text),
            "cleaned_word_count": _count_words(display_text),
        },
    }
