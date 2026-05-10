"""Model configuration placeholders for future summarization engines."""

from __future__ import annotations


# TODO: Add TF-IDF configuration values.
TF_IDF_CONFIG: dict[str, int | float | str] = {}

# TODO: Add TextRank configuration values.
TEXTRANK_CONFIG: dict[str, int | float | str] = {}

# TODO: Add Transformer model names and generation settings.
TRANSFORMER_CONFIG: dict[str, int | float | str] = {}

# Production English summarization model. Smaller models should only be used for local debugging.
ENGLISH_SUMMARIZATION_MODEL = "facebook/bart-large-cnn"

# TODO: Turkish model will be implemented in a later step.
TURKISH_SUMMARIZATION_MODEL = "mukayese/mt5-base-turkish-summarization"
