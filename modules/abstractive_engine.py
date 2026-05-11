"""Transformer-based abstractive summarization engines."""

from __future__ import annotations

import html
import re
from collections import Counter
from functools import lru_cache
from typing import Any

from config.model_config import ENGLISH_SUMMARIZATION_MODEL, TURKISH_SUMMARIZATION_MODEL
from modules.extractive_engine import summarize_with_textrank
from modules.chunking import chunk_text_by_words
from utils.stopwords import get_stopwords


def _count_words(text: str) -> int:
    """Count whitespace-separated words in text."""
    return len(text.split())


def _empty_transformer_response(
    model_name: str,
    language: str = "en",
    input_word_count: int = 0,
    error: str | None = None,
    chunk_errors: list[str] | None = None,
) -> dict[str, Any]:
    """Build a consistent empty Transformer response."""
    response = {
        "method": "Transformer",
        "language": language,
        "model_name": model_name,
        "summary": "",
        "chunk_count": 0,
        "chunk_summaries": [],
        "input_word_count": input_word_count,
        "summary_word_count": 0,
        "error": error,
        "warning": None,
        "chunk_errors": chunk_errors or [],
        "rejected_chunk_summaries": [],
        "source_filtered_sentences": [],
    }
    return response


def _load_summarizer_pipeline(model_name: str):
    """Load a Hugging Face summarization pipeline for a model name."""
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


@lru_cache(maxsize=1)
def _load_english_summarizer(model_name: str):
    """Load and cache the Hugging Face English summarization pipeline."""
    return _load_summarizer_pipeline(model_name)


@lru_cache(maxsize=1)
def load_turkish_seq2seq_model() -> tuple[Any, Any]:
    """Load and cache the Turkish tokenizer and Seq2Seq model."""
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError as error:
        raise ImportError(
            "Missing dependency: transformers is not installed. "
            "Install project requirements with `pip install -r requirements.txt`."
        ) from error

    tokenizer = AutoTokenizer.from_pretrained(TURKISH_SUMMARIZATION_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(TURKISH_SUMMARIZATION_MODEL)
    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None and hasattr(generation_config, "max_new_tokens"):
        generation_config.max_new_tokens = None

    model_config = getattr(model, "config", None)
    if model_config is not None and hasattr(model_config, "max_new_tokens"):
        model_config.max_new_tokens = None

    model.eval()
    return tokenizer, model


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


def _summarize_turkish_chunk(tokenizer: Any, model: Any, chunk: str) -> str:
    """Summarize a Turkish text chunk with explicit Seq2Seq generation."""
    inputs = tokenizer(
        chunk,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    model_device = getattr(model, "device", None)
    if model_device is not None:
        inputs = {
            name: tensor.to(model_device)
            for name, tensor in inputs.items()
        }

    output_ids = model.generate(
        **inputs,
        max_length=150,
        min_length=40,
        do_sample=False,
        num_beams=4,
        no_repeat_ngram_size=3,
        repetition_penalty=1.25,
        length_penalty=1.0,
        early_stopping=True,
    )

    if output_ids is None or len(output_ids) == 0:
        return ""

    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()


def _summarize_turkish_hybrid_chunk(
    tokenizer: Any,
    model: Any,
    chunk: str,
) -> str:
    """Summarize a TextRank-reduced Turkish chunk with explicit generation."""
    inputs = tokenizer(
        chunk,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    model_device = getattr(model, "device", None)
    if model_device is not None:
        inputs = {
            name: tensor.to(model_device)
            for name, tensor in inputs.items()
        }

    output_ids = model.generate(
        **inputs,
        max_length=150,
        min_length=40,
        do_sample=False,
        num_beams=4,
        no_repeat_ngram_size=3,
        repetition_penalty=1.25,
        length_penalty=1.0,
        early_stopping=True,
    )

    if output_ids is None or len(output_ids) == 0:
        return ""

    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()


def _has_excessive_numbered_pattern(summary: str) -> bool:
    """Return whether a summary is dominated by repeated numbered fragments."""
    numbered_matches = re.findall(r"\b\d+\s*[\.)]", summary)
    if len(numbered_matches) >= 4:
        return True

    compact_summary = re.sub(r"\s+", "", summary)
    return bool(re.search(r"1[\.)],?2[\.)],?3[\.)],?4[\.)]", compact_summary))


def _has_too_many_repeated_phrases(summary: str) -> bool:
    """Return whether a summary repeats short phrases excessively."""
    normalized_words = re.findall(r"\w+", summary.lower(), flags=re.UNICODE)
    if len(normalized_words) < 20:
        return False

    for phrase_length in range(2, 6):
        phrases = [
            " ".join(normalized_words[index : index + phrase_length])
            for index in range(len(normalized_words) - phrase_length + 1)
        ]
        phrase_counts = Counter(phrases)
        if any(count > 3 for count in phrase_counts.values()):
            return True

    return False


def _is_mostly_numbers_or_punctuation(text: str) -> bool:
    """Return whether text contains too little alphabetic content."""
    stripped_text = text.strip()
    if not stripped_text:
        return True

    alphabetic_count = sum(character.isalpha() for character in stripped_text)
    return alphabetic_count / max(len(stripped_text), 1) < 0.35


def _normalize_turkish_for_rules(text: str) -> str:
    """Normalize Turkish text for simple rule-based filtering."""
    normalized_text = text.lower().replace("i̇", "i")
    return re.sub(r"\s+", " ", normalized_text).strip()


def is_valid_transformer_summary(summary: str, language: str) -> bool:
    """Return whether a Transformer summary is usable for display and combination."""
    normalized_summary = " ".join(summary.split())
    if not normalized_summary:
        return False

    min_word_count = 8 if language == "tr" else 20
    if _count_words(normalized_summary) < min_word_count:
        return False
    if _has_excessive_numbered_pattern(normalized_summary):
        return False
    if _has_too_many_repeated_phrases(normalized_summary):
        return False
    if _is_mostly_numbers_or_punctuation(normalized_summary):
        return False

    return True


def _clean_html_artifacts(text: str) -> str:
    """Remove HTML entities and tag fragments from generated summaries."""
    cleaned_text = html.unescape(text)
    cleaned_text = re.sub(r"<br\s*/?\s*>", " ", cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r"<br\s*/?\s*\.?>", " ", cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r"</?\w+[^>]*>", " ", cleaned_text)
    cleaned_text = re.sub(r"&nbsp;", " ", cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r"<[^>\s]*(?:\s|$)", " ", cleaned_text)
    cleaned_text = re.sub(r"/>\s*", " ", cleaned_text)
    cleaned_text = re.sub(r"\s*/\s*\.", ".", cleaned_text)
    return re.sub(r"\s+", " ", cleaned_text).strip()


def _remove_citations(text: str) -> str:
    """Remove common parenthetical author-year citations from summaries."""
    citation_patterns = [
        r"\([A-ZÇĞİÖŞÜa-zçğıöşü.'\-]+\s+\d{4}\s*:\s*\d+\s*[-–]\s*\d+\s*\)",
        r"\([A-ZÇĞİÖŞÜa-zçğıöşü.'\-]+\s+\d{4}[^)]*\)",
        r"\([A-ZÇĞİÖŞÜa-zçğıöşü.'\-]+\s*,\s*\d{4}[^)]*\)",
    ]
    cleaned_text = text
    for pattern in citation_patterns:
        cleaned_text = re.sub(pattern, " ", cleaned_text)

    return re.sub(r"\s+", " ", cleaned_text).strip()


def _split_transformer_summary_sentences(text: str) -> list[str]:
    """Split a generated summary into display-ready sentence candidates."""
    normalized_text = re.sub(r"\s+", " ", text).strip()
    if not normalized_text:
        return []

    sentence_matches = re.findall(r"[^.!?]+(?:[.!?]+|$)", normalized_text)
    return [sentence.strip() for sentence in sentence_matches if sentence.strip()]


def _normalize_sentence_for_comparison(sentence: str) -> str:
    """Normalize a sentence for duplicate detection."""
    normalized_sentence = sentence.lower()
    normalized_sentence = re.sub(r"[^\wçğıöşüÇĞİÖŞÜ]+", " ", normalized_sentence)
    return re.sub(r"\s+", " ", normalized_sentence).strip()


def _normalize_token(token: str, language: str) -> str:
    """Normalize one token with small language-aware handling."""
    normalized_token = token.lower()
    if language.lower().strip() == "tr":
        normalized_token = normalized_token.replace("i̇", "i")

    return normalized_token


def _important_tokens(text: str, language: str) -> list[str]:
    """Extract simple content-bearing tokens for source overlap checks."""
    normalized_language = language.lower().strip()
    stopwords = {
        _normalize_token(stopword, normalized_language)
        for stopword in get_stopwords(normalized_language)
    }
    raw_tokens = re.findall(r"[A-Za-zÇĞİÖŞÜçğıöşü]+", text)
    tokens = []

    for raw_token in raw_tokens:
        token = _normalize_token(raw_token, normalized_language)
        if len(token) < 3:
            continue
        if not token.isalpha():
            continue
        if token in stopwords:
            continue

        tokens.append(token)

    return tokens


def _deduplicate_sentences(sentences: list[str]) -> list[str]:
    """Remove duplicate sentences while preserving their original order."""
    deduplicated_sentences = []
    seen_sentences = set()

    for sentence in sentences:
        comparison_key = _normalize_sentence_for_comparison(sentence)
        if not comparison_key or comparison_key in seen_sentences:
            continue

        seen_sentences.add(comparison_key)
        deduplicated_sentences.append(sentence)

    return deduplicated_sentences


def _postprocess_english_transformer_summary(summary: str) -> str:
    """Lightly clean English Transformer summaries without aggressive filtering."""
    normalized_summary = re.sub(r"\s+", " ", summary).strip()
    if not normalized_summary:
        return ""

    sentences = [
        re.sub(r"\s+", " ", sentence).strip()
        for sentence in _split_transformer_summary_sentences(normalized_summary)
    ]
    sentences = [sentence for sentence in sentences if sentence]

    if not sentences:
        return normalized_summary

    return " ".join(_deduplicate_sentences(sentences))


def _polish_turkish_sentence(sentence: str) -> str:
    """Apply small display-level fixes to a kept Turkish sentence."""
    polished_sentence = re.sub(r"\s+", " ", sentence).strip()
    if not polished_sentence:
        return ""

    first_character = polished_sentence[0]
    if first_character.islower():
        polished_sentence = f"{first_character.upper()}{polished_sentence[1:]}"

    return polished_sentence


def _prepare_turkish_summary_sentence(sentence: str) -> str:
    """Apply light, non-destructive cleanup to one generated Turkish sentence."""
    prepared_sentence = sentence.strip(" -•\t")
    prepared_sentence = re.sub(r"https?://\S+|www\.\S+", " ", prepared_sentence)
    prepared_sentence = re.sub(r"\S+@\S+", " ", prepared_sentence)
    prepared_sentence = re.sub(r":\s*(?:İş|Iş|Is)\.?\s*$", ".", prepared_sentence)
    prepared_sentence = re.sub(r"\b(?:İş|Iş|Is)\.\s*$", "", prepared_sentence)
    prepared_sentence = re.sub(r"\.{2,}", ".", prepared_sentence)
    prepared_sentence = re.sub(r"\s+([,.!?;:])", r"\1", prepared_sentence)
    prepared_sentence = re.sub(r"\s+", " ", prepared_sentence).strip()

    if not prepared_sentence:
        return ""
    if prepared_sentence[-1] not in ".!?":
        prepared_sentence = f"{prepared_sentence}."

    return _polish_turkish_sentence(prepared_sentence)


def _is_hard_bad_turkish_sentence(sentence: str) -> bool:
    """Return whether a sentence is definitely trash and should not be restored."""
    stripped_sentence = sentence.strip()
    normalized_sentence = _normalize_turkish_for_rules(stripped_sentence)

    if not stripped_sentence:
        return True
    if normalized_sentence in {"iş.", "is.", "işte.", "/>"}:
        return True
    if (
        "/>" in stripped_sentence
        or "<br" in normalized_sentence
        or "</" in normalized_sentence
    ):
        return True
    if "fifty shades" in normalized_sentence:
        return True
    if "http://" in normalized_sentence or "https://" in normalized_sentence:
        return True
    if "www." in normalized_sentence or "@" in normalized_sentence:
        return True
    if "bilmeniz gerekenler" in normalized_sentence:
        return True
    if "işte detaylar" in normalized_sentence or "iste detaylar" in normalized_sentence:
        return True
    if "ve kapsamında" in normalized_sentence:
        return True
    if (
        "ve tarafından desteklenen erzurum, erzincan ve kapsamında"
        in normalized_sentence
    ):
        return True
    if normalized_sentence.endswith((" gün.", " yayında.", " yolu.")):
        return True

    return False


def _is_obvious_bad_turkish_sentence(sentence: str) -> bool:
    """Return whether a generated sentence is clearly unsuitable for final output."""
    normalized_sentence = _normalize_turkish_for_rules(sentence)
    word_count = _count_words(sentence)

    if _is_hard_bad_turkish_sentence(sentence):
        return True
    if word_count < 4:
        return True
    if normalized_sentence.startswith(("işte ", "iste ")) and word_count <= 10:
        return True
    if normalized_sentence.startswith(("işte ", "iste ")) and any(
        phrase in normalized_sentence
        for phrase in (
            "genel bir değerlendirme",
            "maddeler arasında",
            "tarih müfredatı",
            "geliştirilen bu yöntem",
            "tarih öğretim programları",
        )
    ):
        return True

    return False


def _deduplicate_turkish_sentences(sentences: list[str]) -> list[str]:
    """Remove duplicate Turkish summary sentences while preserving order."""
    return _deduplicate_sentences(sentences)


def _target_turkish_summary_min_words(
    input_word_count: int,
    summary_ratio: float | None,
) -> int:
    """Estimate a practical lower bound for Turkish Transformer summaries."""
    if input_word_count <= 0 or summary_ratio is None:
        return 0

    target_word_count = int(input_word_count * summary_ratio * 0.15)
    return min(260, max(80, target_word_count))


def _repair_generated_turkish_text(text: str) -> str:
    """Repair common PDF/model artifacts in generated Turkish summaries."""
    repaired_text = _clean_html_artifacts(text)
    repaired_text = _remove_citations(repaired_text)
    artifact_replacements = {
        "gerçek-leştiril": "gerçekleştiril",
        "gerçek-leştir": "gerçekleştir",
        "çağ-cıl": "çağcıl",
        "öğren-cilerin": "öğrencilerin",
        "Yurttaşlık Duygusunun Gelişimini Sağlar": (
            "yurttaşlık duygusunun gelişimini sağlar"
        ),
        "İnsanlığın Kültürel Mirasının Kavranmasını": (
            "insanlığın kültürel mirasının kavranmasını"
        ),
        "Değişimleri Anlamamızı": "değişimleri anlamamızı",
    }

    for artifact, replacement in artifact_replacements.items():
        repaired_text = repaired_text.replace(artifact, replacement)

    repaired_text = re.sub(r"\s+([,.!?;:])", r"\1", repaired_text)
    repaired_text = re.sub(r"([.!?]){2,}", r"\1", repaired_text)
    return re.sub(r"\s+", " ", repaired_text).strip()


def postprocess_turkish_transformer_summary(
    summary: str,
    min_sentences: int = 5,
) -> str:
    """Lightly clean, deduplicate, and paragraphize Turkish Transformer summaries.

    The function keeps a fallback sentence list so post-processing does not
    collapse a long generated summary into one or two over-filtered sentences.
    """
    normalized_summary = _repair_generated_turkish_text(summary)
    if not normalized_summary:
        return ""

    fallback_sentences = []
    cleaned_sentences = []

    for sentence in _split_transformer_summary_sentences(normalized_summary):
        sentence = _prepare_turkish_summary_sentence(sentence)
        if not sentence:
            continue

        if _count_words(sentence) < 6:
            continue
        if _is_mostly_numbers_or_punctuation(sentence):
            continue
        if _is_hard_bad_turkish_sentence(sentence):
            continue

        fallback_sentences.append(sentence)
        if not _is_obvious_bad_turkish_sentence(sentence):
            cleaned_sentences.append(sentence)

    fallback_sentences = _deduplicate_turkish_sentences(fallback_sentences)
    cleaned_sentences = _deduplicate_turkish_sentences(cleaned_sentences)

    if len(cleaned_sentences) < min_sentences and fallback_sentences:
        cleaned_sentences = fallback_sentences

    if not cleaned_sentences:
        return normalized_summary

    return " ".join(cleaned_sentences)


def postprocess_transformer_summary(summary: str, language: str) -> str:
    """Post-process Transformer summaries using language-specific light rules."""
    normalized_language = language.lower().strip()

    if normalized_language == "tr":
        return postprocess_turkish_transformer_summary(summary)
    if normalized_language == "en":
        return _postprocess_english_transformer_summary(summary)

    return re.sub(r"\s+", " ", summary).strip()


def _filter_summary_sentences_by_source_overlap(
    summary: str,
    source_text: str,
    language: str,
    min_overlap_ratio: float = 0.30,
) -> tuple[str, list[str]]:
    """Return source-grounded summary text and removed low-overlap sentences."""
    source_tokens = set(_important_tokens(source_text, language))
    kept_sentences = []
    removed_sentences = []

    for sentence in _split_transformer_summary_sentences(summary):
        cleaned_sentence = re.sub(r"\s+", " ", sentence).strip()
        if not cleaned_sentence:
            continue

        sentence_tokens = _important_tokens(cleaned_sentence, language)
        if len(sentence_tokens) < 5:
            kept_sentences.append(cleaned_sentence)
            continue

        overlap_count = sum(token in source_tokens for token in sentence_tokens)
        overlap_ratio = overlap_count / len(sentence_tokens)

        if overlap_ratio < min_overlap_ratio:
            removed_sentences.append(cleaned_sentence)
            continue

        kept_sentences.append(cleaned_sentence)

    return " ".join(kept_sentences), removed_sentences


def filter_summary_sentences_by_source_overlap(
    summary: str,
    source_text: str,
    language: str,
    min_overlap_ratio: float = 0.30,
) -> str:
    """Filter low source-overlap summary sentences while preserving order."""
    filtered_summary, _ = _filter_summary_sentences_by_source_overlap(
        summary,
        source_text,
        language,
        min_overlap_ratio,
    )

    return filtered_summary


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
            language="en",
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
            language="en",
            input_word_count=input_word_count,
            error="No chunks could be created from the input text.",
        )

    try:
        summarizer = _load_english_summarizer(ENGLISH_SUMMARIZATION_MODEL)
    except ImportError as error:
        return _empty_transformer_response(
            model_name=ENGLISH_SUMMARIZATION_MODEL,
            language="en",
            input_word_count=input_word_count,
            error=str(error),
        )
    except Exception as error:
        return _empty_transformer_response(
            model_name=ENGLISH_SUMMARIZATION_MODEL,
            language="en",
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
            language="en",
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

    final_summary = postprocess_transformer_summary(final_summary, "en")

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
        "warning": None,
        "chunk_errors": chunk_errors,
    }

    return response


def summarize_turkish_transformer(
    text: str,
    max_words_per_chunk: int = 250,
) -> dict[str, Any]:
    """Summarize Turkish academic text with a chunk-based Transformer pipeline.

    Args:
        text: Cleaned Turkish article text.
        max_words_per_chunk: Maximum words per chunk before model summarization.

    Returns:
        Dictionary containing the final summary, chunk summaries, model metadata,
        errors, and word counts.
    """
    normalized_text = text.strip()
    input_word_count = _count_words(normalized_text)

    if not normalized_text:
        return _empty_transformer_response(
            model_name=TURKISH_SUMMARIZATION_MODEL,
            language="tr",
            input_word_count=0,
        )

    chunks = chunk_text_by_words(
        normalized_text,
        max_words=max_words_per_chunk,
        overlap_words=40,
    )

    if not chunks:
        return _empty_transformer_response(
            model_name=TURKISH_SUMMARIZATION_MODEL,
            language="tr",
            input_word_count=input_word_count,
            error="No chunks could be created from the input text.",
        )

    try:
        tokenizer, model = load_turkish_seq2seq_model()
    except ImportError as error:
        return _empty_transformer_response(
            model_name=TURKISH_SUMMARIZATION_MODEL,
            language="tr",
            input_word_count=input_word_count,
            error=str(error),
        )
    except Exception as error:
        return _empty_transformer_response(
            model_name=TURKISH_SUMMARIZATION_MODEL,
            language="tr",
            input_word_count=input_word_count,
            error=f"Turkish Transformer model loading failed: {error}",
        )

    chunk_summaries = []
    rejected_chunk_summaries = []
    generated_chunk_summaries = []
    chunk_errors = []

    for chunk_index, chunk in enumerate(chunks, start=1):
        try:
            chunk_summary = _summarize_turkish_chunk(tokenizer, model, chunk)
        except Exception as error:
            chunk_errors.append(f"Chunk {chunk_index}: {error}")
            continue

        if not chunk_summary:
            continue

        chunk_summary = postprocess_transformer_summary(chunk_summary, "tr")
        if not chunk_summary:
            continue

        generated_chunk_summaries.append(chunk_summary)
        if is_valid_transformer_summary(chunk_summary, "tr"):
            chunk_summaries.append(chunk_summary)
        else:
            rejected_chunk_summaries.append(chunk_summary)

    if not generated_chunk_summaries:
        return _empty_transformer_response(
            model_name=TURKISH_SUMMARIZATION_MODEL,
            language="tr",
            input_word_count=input_word_count,
            error="Turkish Transformer summarization failed for all chunks.",
            chunk_errors=chunk_errors,
        )

    warning = None
    if chunk_summaries:
        final_summary = postprocess_transformer_summary(
            " ".join(chunk_summaries),
            "tr",
        )
    else:
        final_summary = postprocess_transformer_summary(
            max(generated_chunk_summaries, key=_count_words),
            "tr",
        )
        warning = (
            "All Turkish chunk summaries were rejected by the quality filter; "
            "showing the least bad available chunk summary."
        )

    return {
        "method": "Transformer",
        "language": "tr",
        "model_name": TURKISH_SUMMARIZATION_MODEL,
        "summary": final_summary,
        "chunk_count": len(chunks),
        "chunk_summaries": chunk_summaries,
        "input_word_count": input_word_count,
        "summary_word_count": _count_words(final_summary),
        "error": None,
        "warning": warning,
        "chunk_errors": chunk_errors,
        "rejected_chunk_summaries": rejected_chunk_summaries,
    }


def summarize_turkish_hybrid_transformer(
    text: str,
    extractive_ratio: float = 0.30,
    max_words_per_chunk: int = 300,
) -> dict[str, Any]:
    """Summarize Turkish text with TextRank reduction followed by Transformer.

    Args:
        text: Cleaned Turkish article text.
        extractive_ratio: Fraction of valid TextRank sentences used as Transformer input.
        max_words_per_chunk: Maximum words per reduced chunk.

    Returns:
        Hybrid Transformer summary result with reduction and chunk metadata.
    """
    normalized_text = text.strip()
    input_word_count = _count_words(normalized_text)

    if not normalized_text:
        response = _empty_transformer_response(
            model_name=TURKISH_SUMMARIZATION_MODEL,
            language="tr",
            input_word_count=0,
        )
        response["method"] = "Hybrid Transformer"
        response["reduced_input_word_count"] = 0
        response["extractive_ratio"] = extractive_ratio
        response["source_filtered_sentences"] = []
        return response

    textrank_result = summarize_with_textrank(
        normalized_text,
        summary_ratio=extractive_ratio,
    )
    reduced_text = str(textrank_result.get("summary", "")).strip()
    warning = None

    if not reduced_text:
        reduced_text = normalized_text
        warning = (
            "TextRank could not produce reduced Turkish input; "
            "using cleaned text for hybrid Transformer summarization."
        )

    reduced_input_word_count = _count_words(reduced_text)
    chunks = chunk_text_by_words(
        reduced_text,
        max_words=max_words_per_chunk,
        overlap_words=40,
    )

    if not chunks:
        response = _empty_transformer_response(
            model_name=TURKISH_SUMMARIZATION_MODEL,
            language="tr",
            input_word_count=input_word_count,
            error="No chunks could be created from the TextRank-reduced input.",
        )
        response["method"] = "Hybrid Transformer"
        response["reduced_input_word_count"] = reduced_input_word_count
        response["extractive_ratio"] = extractive_ratio
        response["warning"] = warning
        response["source_filtered_sentences"] = []
        return response

    try:
        tokenizer, model = load_turkish_seq2seq_model()
    except ImportError as error:
        response = _empty_transformer_response(
            model_name=TURKISH_SUMMARIZATION_MODEL,
            language="tr",
            input_word_count=input_word_count,
            error=str(error),
        )
        response["method"] = "Hybrid Transformer"
        response["reduced_input_word_count"] = reduced_input_word_count
        response["extractive_ratio"] = extractive_ratio
        response["warning"] = warning
        response["source_filtered_sentences"] = []
        return response
    except Exception as error:
        response = _empty_transformer_response(
            model_name=TURKISH_SUMMARIZATION_MODEL,
            language="tr",
            input_word_count=input_word_count,
            error=f"Turkish Hybrid Transformer model loading failed: {error}",
        )
        response["method"] = "Hybrid Transformer"
        response["reduced_input_word_count"] = reduced_input_word_count
        response["extractive_ratio"] = extractive_ratio
        response["warning"] = warning
        response["source_filtered_sentences"] = []
        return response

    chunk_summaries = []
    rejected_chunk_summaries = []
    generated_chunk_summaries = []
    chunk_errors = []

    for chunk_index, chunk in enumerate(chunks, start=1):
        try:
            chunk_summary = _summarize_turkish_hybrid_chunk(
                tokenizer,
                model,
                chunk,
            )
        except Exception as error:
            chunk_errors.append(f"Chunk {chunk_index}: {error}")
            continue

        if not chunk_summary:
            continue

        chunk_summary = postprocess_transformer_summary(chunk_summary, "tr")
        if not chunk_summary:
            continue

        generated_chunk_summaries.append(chunk_summary)
        if is_valid_transformer_summary(chunk_summary, "tr"):
            chunk_summaries.append(chunk_summary)
        else:
            rejected_chunk_summaries.append(chunk_summary)

    if not generated_chunk_summaries:
        response = _empty_transformer_response(
            model_name=TURKISH_SUMMARIZATION_MODEL,
            language="tr",
            input_word_count=input_word_count,
            error="Turkish Hybrid Transformer summarization failed for all chunks.",
            chunk_errors=chunk_errors,
        )
        response["method"] = "Hybrid Transformer"
        response["chunk_count"] = len(chunks)
        response["reduced_input_word_count"] = reduced_input_word_count
        response["extractive_ratio"] = extractive_ratio
        response["warning"] = warning
        response["source_filtered_sentences"] = []
        return response

    if chunk_summaries:
        final_summary = postprocess_transformer_summary(
            " ".join(chunk_summaries),
            "tr",
        )
    else:
        final_summary = postprocess_transformer_summary(
            max(generated_chunk_summaries, key=_count_words),
            "tr",
        )
        warning = (
            "All Turkish hybrid chunk summaries were rejected by the quality filter; "
            "showing the least bad available chunk summary."
        )

    target_min_words = _target_turkish_summary_min_words(input_word_count, 0.20)
    if target_min_words and _count_words(final_summary) < target_min_words:
        fallback_summary = postprocess_transformer_summary(
            " ".join(generated_chunk_summaries),
            "tr",
        )
        if _count_words(fallback_summary) > _count_words(final_summary):
            final_summary = fallback_summary
            fallback_warning = (
                "Turkish summary post-processing was relaxed because the strict "
                "result was shorter than the target minimum word count."
            )
            warning = f"{warning} {fallback_warning}" if warning else fallback_warning

    filtered_summary, source_filtered_sentences = (
        _filter_summary_sentences_by_source_overlap(
            final_summary,
            reduced_text,
            "tr",
            min_overlap_ratio=0.30,
        )
    )
    if filtered_summary != final_summary:
        final_summary = postprocess_transformer_summary(filtered_summary, "tr")

    if not final_summary and source_filtered_sentences:
        warning_text = "Source-overlap filtering removed every final sentence."
        warning = f"{warning} {warning_text}" if warning else warning_text

    return {
        "method": "Hybrid Transformer",
        "language": "tr",
        "model_name": TURKISH_SUMMARIZATION_MODEL,
        "summary": final_summary,
        "chunk_count": len(chunks),
        "chunk_summaries": chunk_summaries,
        "rejected_chunk_summaries": rejected_chunk_summaries,
        "source_filtered_sentences": source_filtered_sentences,
        "input_word_count": input_word_count,
        "reduced_input_word_count": reduced_input_word_count,
        "summary_word_count": _count_words(final_summary),
        "extractive_ratio": extractive_ratio,
        "error": None,
        "warning": warning,
        "chunk_errors": chunk_errors,
    }


def summarize_with_transformer(text: str, language: str) -> dict[str, Any]:
    """Route Transformer summarization by language."""
    if language == "en":
        return summarize_english_transformer(text)
    if language == "tr":
        return summarize_turkish_hybrid_transformer(text)

    return {
        "method": "Transformer",
        "language": language,
        "model_name": "",
        "summary": "",
        "chunk_count": 0,
        "chunk_summaries": [],
        "input_word_count": _count_words(text),
        "summary_word_count": 0,
        "error": "Transformer summarization requires detected language to be English or Turkish.",
        "warning": None,
        "chunk_errors": [],
        "rejected_chunk_summaries": [],
        "source_filtered_sentences": [],
    }
