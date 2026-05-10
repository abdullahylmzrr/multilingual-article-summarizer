"""English article processing pipeline."""

from __future__ import annotations

from modules.preprocessing import preprocess_text


def prepare_english_text(text: str) -> dict[str, object]:
    """Prepare English academic text for future summarization."""
    # TODO: Add English-specific tokenization, stopword handling, and normalization.
    return preprocess_text(text, "en")
