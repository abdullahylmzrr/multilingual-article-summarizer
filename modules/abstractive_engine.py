"""Transformer-based abstractive summarization engines."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from config.model_config import ENGLISH_SUMMARIZATION_MODEL
from modules.chunking import chunk_text_by_words


def _count_words(text: str) -> int:
    """Count whitespace-separated words in text."""
    return len(text.split())


def _empty_transformer_response(
    model_name: str,
    input_word_count: int = 0,
    error: str | None = None,
    chunk_errors: list[str] | None = None,
) -> dict[str, Any]:
    """Build a consistent empty Transformer response."""
    response = {
        "method": "Transformer",
        "language": "en",
        "model_name": model_name,
        "summary": "",
        "chunk_count": 0,
        "chunk_summaries": [],
        "input_word_count": input_word_count,
        "summary_word_count": 0,
        "error": error,
        "chunk_errors": chunk_errors or [],
    }
    return response


@lru_cache(maxsize=1)
def _load_english_summarizer(model_name: str):
    """Load and cache the Hugging Face English summarization pipeline."""
    try:
        from transformers import pipeline
    except ImportError as error:
        raise ImportError(
            "Missing dependency: transformers is not installed. "
            "Install project requirements with `pip install -r requirements.txt`."
        ) from error

    return pipeline(
        task="summarization",
        model=model_name,
        tokenizer=model_name,
    )


def _summary_lengths_for_chunk(chunk: str) -> tuple[int, int]:
    """Choose conservative max/min summary lengths for a chunk."""
    word_count = _count_words(chunk)
    max_length = min(180, max(40, int(word_count * 0.55)))
    min_length = min(60, max(15, int(word_count * 0.15)))

    if min_length >= max_length:
        min_length = max(5, max_length // 2)

    return max_length, min_length


def _summarize_chunk(summarizer, chunk: str) -> str:
    """Summarize a single text chunk with a Hugging Face pipeline."""
    max_length, min_length = _summary_lengths_for_chunk(chunk)
    result = summarizer(
        chunk,
        max_length=max_length,
        min_length=min_length,
        do_sample=False,
        truncation=True,
    )

    if not result:
        return ""

    return str(result[0].get("summary_text", "")).strip()


def summarize_english_transformer(
    text: str,
    max_words_per_chunk: int = 450,
) -> dict[str, Any]:
    """Summarize English academic text with a chunk-based Transformer pipeline.

    Args:
        text: Cleaned English article text.
        max_words_per_chunk: Maximum words per chunk before model summarization.

    Returns:
        Dictionary containing the final summary, chunk summaries, model metadata,
        and word counts.
    """
    normalized_text = text.strip()
    input_word_count = _count_words(normalized_text)

    if not normalized_text:
        return _empty_transformer_response(
            model_name=ENGLISH_SUMMARIZATION_MODEL,
            input_word_count=0,
        )

    chunks = chunk_text_by_words(
        normalized_text,
        max_words=max_words_per_chunk,
        overlap_words=50,
    )

    if not chunks:
        return _empty_transformer_response(
            model_name=ENGLISH_SUMMARIZATION_MODEL,
            input_word_count=input_word_count,
            error="No chunks could be created from the input text.",
        )

    try:
        summarizer = _load_english_summarizer(ENGLISH_SUMMARIZATION_MODEL)
    except ImportError as error:
        return _empty_transformer_response(
            model_name=ENGLISH_SUMMARIZATION_MODEL,
            input_word_count=input_word_count,
            error=str(error),
        )
    except Exception as error:
        return _empty_transformer_response(
            model_name=ENGLISH_SUMMARIZATION_MODEL,
            input_word_count=input_word_count,
            error=f"Transformer model loading failed: {error}",
        )

    chunk_summaries = []
    chunk_errors = []

    for chunk_index, chunk in enumerate(chunks, start=1):
        try:
            chunk_summary = _summarize_chunk(summarizer, chunk)
        except Exception as error:
            chunk_errors.append(f"Chunk {chunk_index}: {error}")
            continue

        if chunk_summary:
            chunk_summaries.append(chunk_summary)

    if not chunk_summaries:
        return _empty_transformer_response(
            model_name=ENGLISH_SUMMARIZATION_MODEL,
            input_word_count=input_word_count,
            error="Transformer summarization failed for all chunks.",
            chunk_errors=chunk_errors,
        )

    combined_summary = " ".join(chunk_summaries).strip()
    final_summary = combined_summary

    if _count_words(combined_summary) > max_words_per_chunk:
        try:
            final_summary = _summarize_chunk(summarizer, combined_summary)
        except Exception as error:
            chunk_errors.append(f"Final summary pass: {error}")
            final_summary = combined_summary

    response = {
        "method": "Transformer",
        "language": "en",
        "model_name": ENGLISH_SUMMARIZATION_MODEL,
        "summary": final_summary,
        "chunk_count": len(chunks),
        "chunk_summaries": chunk_summaries,
        "input_word_count": input_word_count,
        "summary_word_count": _count_words(final_summary),
        "error": None,
        "chunk_errors": chunk_errors,
    }

    return response


def summarize_with_transformer(text: str, language: str) -> dict[str, Any]:
    """Route Transformer summarization by language."""
    if language == "en":
        return summarize_english_transformer(text)

    return {
        "method": "Transformer",
        "language": language,
        "model_name": "",
        "summary": "",
        "chunk_count": 0,
        "chunk_summaries": [],
        "input_word_count": _count_words(text),
        "summary_word_count": 0,
        "error": "Transformer summarization is currently implemented only for English.",
        "chunk_errors": [],
    }
