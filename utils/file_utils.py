"""File-system helper utilities."""

from __future__ import annotations

from pathlib import Path


def ensure_directory(path: str | Path) -> Path:
    """Create a directory if needed and return it as a Path object."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def is_pdf_file(filename: str) -> bool:
    """Return whether a filename appears to be a PDF."""
    return filename.lower().endswith(".pdf")
