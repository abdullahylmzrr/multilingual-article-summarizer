"""General application settings."""

from __future__ import annotations

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SAMPLE_PDF_DIR = DATA_DIR / "sample_pdfs"
OUTPUT_DIR = DATA_DIR / "outputs"
MODEL_DIR = BASE_DIR / "models" / "downloaded_models"

SUPPORTED_LANGUAGES = {"tr", "en"}
TEXT_PREVIEW_LIMIT = 3000
