"""Turkish article processing pipeline."""

from __future__ import annotations

from modules.preprocessing import preprocess_text


def prepare_turkish_text(text: str) -> dict[str, object]:
    """Prepare Turkish academic text for future summarization."""
    # TODO: Add Turkish-specific tokenization, stopword handling, and normalization.
    return preprocess_text(text, "tr")
