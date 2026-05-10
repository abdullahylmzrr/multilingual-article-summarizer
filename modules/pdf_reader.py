"""PDF text extraction utilities."""

from __future__ import annotations

from pathlib import Path

import fitz


class PDFReadError(RuntimeError):
    """Raised when text cannot be extracted from a PDF."""


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from a PDF represented as bytes.

    Args:
        file_bytes: Raw PDF file content.

    Returns:
        Extracted text from all pages, joined by blank lines.

    Raises:
        PDFReadError: If the PDF cannot be opened or read.
    """
    if not file_bytes:
        raise PDFReadError("The uploaded PDF file is empty.")

    try:
        with fitz.open(stream=file_bytes, filetype="pdf") as document:
            pages = [page.get_text("text") for page in document]
    except Exception as error:
        raise PDFReadError("Could not read the uploaded PDF file.") from error

    return "\n\n".join(page.strip() for page in pages if page.strip())


def extract_text_from_path(pdf_path: str | Path) -> str:
    """Extract text from a PDF file path.

    Args:
        pdf_path: Path to a local PDF file.

    Returns:
        Extracted text from all pages.

    Raises:
        PDFReadError: If the file is missing or cannot be read.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise PDFReadError(f"PDF file not found: {path}")

    try:
        return extract_text_from_pdf(path.read_bytes())
    except OSError as error:
        raise PDFReadError(f"Could not open PDF file: {path}") from error
