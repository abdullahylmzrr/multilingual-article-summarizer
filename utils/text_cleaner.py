"""Text cleaning helpers."""

from __future__ import annotations

import re


def normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace while preserving readable text flow."""
    return re.sub(r"\s+", " ", text).strip()
