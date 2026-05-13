"""Multilingual preprocessing helpers for future summarization pipelines."""

from __future__ import annotations

import importlib
import re

from utils.stopwords import get_stopwords
import utils.text_cleaner as text_cleaner


TOKEN_CLEANUP_PATTERN = re.compile(r"[^\w\sçğıöşüÇĞİÖŞÜ]", flags=re.UNICODE)
REQUIRED_TEXT_CLEANER_FUNCTIONS = {
    "clean_for_summarization",
    "clean_lines",
    "fix_pdf_hyphenation",
    "remove_academic_pdf_noise",
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
    cleaned_text = cleaner.fix_pdf_hyphenation(raw_text)
    cleaned_text = cleaner.remove_urls(cleaned_text)
    cleaned_text = cleaner.remove_emails(cleaned_text)
    cleaned_text = cleaner.remove_doi_patterns(cleaned_text)
    cleaned_text = cleaner.remove_references_section(cleaned_text, language)
    cleaned_text = cleaner.remove_academic_pdf_noise(cleaned_text)
    cleaned_text = cleaner.clean_lines(cleaned_text)
    return cleaner.remove_extra_whitespace(cleaned_text)


def _build_summarization_text(raw_text: str, language: str) -> dict[str, object]:
    """Create stronger-cleaned text optimised for summarization algorithms.

    Starts from raw, line-preserved text so line-level PDF artifacts can be
    removed before whitespace is collapsed.

    Args:
        raw_text: Raw text extracted from the PDF.
        language: Detected language code (``"tr"`` or ``"en"``).

    Returns:
        Dictionary with ``"text"`` and ``"debug"`` keys.
    """
    cleaner = _get_text_cleaner_module()
    try:
        cleaned_text, debug_counts = cleaner.clean_for_summarization(raw_text, language)
        return {"text": cleaned_text, "debug": debug_counts}
    except Exception:
        return {"text": raw_text, "debug": {}}


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
    """Run the course-style NLP preprocessing pipeline for one article.

    Args:
        raw_text: Raw text extracted from a PDF.
        language: Detected language code, usually ``"tr"`` or ``"en"``.

    Returns:
        Dictionary containing:
        - ``display_text``: Readable cleaned text for UI preview.
        - ``summarization_text``: Stronger-cleaned text for TF-IDF / TextRank / Transformer.
        - ``nlp_text``: Lowercased stopword-filtered text for NLP analysis.
        - ``stats``: Character and word counts plus debug removal counts.
    """
    # Stage 1: Raw PDF text normalization / Ham PDF metni giriş olarak korunur.
    normalized_raw_text = raw_text or ""

    # Stage 2: Display text cleaning for UI preview / Okunabilir önizleme metni.
    display_text = _build_display_text(normalized_raw_text, language)

    # Stage 3: Summarization text cleaning before line collapse.
    # TF-IDF, TextRank ve Transformer için satır yapısı bozulmadan önce
    # header/footer, tablo ve metadata temizliği yapılır.
    summarization_result = _build_summarization_text(normalized_raw_text, language)
    summarization_text = str(summarization_result.get("text", display_text))
    summarization_debug = summarization_result.get("debug", {})

    # Stage 4: NLP text generation with stopword filtering.
    # Bu temsil analiz/önizleme içindir; final özetlerde doğal cümlelerin
    # yerine kullanılmaz.
    nlp_text = _build_nlp_text(summarization_text or display_text, language)

    # Stage 5: Statistics and fallback-friendly metadata.
    # cleaned_* anahtarları eski app kodlarıyla uyumluluk için display_* alias'ıdır.
    display_word_count = _count_words(display_text)
    summarization_word_count = _count_words(summarization_text)

    return {
        "display_text": display_text,
        "summarization_text": summarization_text,
        "nlp_text": nlp_text,
        "stats": {
            # Core counts
            "raw_character_count": len(normalized_raw_text),
            "display_character_count": len(display_text),
            "summarization_character_count": len(summarization_text),
            "nlp_character_count": len(nlp_text),
            "raw_word_count": _count_words(normalized_raw_text),
            "display_word_count": display_word_count,
            "summarization_word_count": summarization_word_count,
            "nlp_word_count": _count_words(nlp_text),
            # Backward compatibility aliases
            "cleaned_character_count": len(display_text),
            "cleaned_word_count": display_word_count,
            # Debug removal counts (optional, from summarization pipeline)
            "removed_metadata_line_count": summarization_debug.get(
                "removed_metadata_line_count", 0
            ),
            "removed_repeated_line_count": summarization_debug.get(
                "removed_repeated_line_count", 0
            ),
            "removed_table_line_count": summarization_debug.get(
                "removed_table_line_count", 0
            ),
            "removed_short_noise_line_count": summarization_debug.get(
                "removed_short_noise_line_count", 0
            ),
            "repaired_hyphenation_count": summarization_debug.get(
                "repaired_hyphenation_count", 0
            ),
        },
    }
