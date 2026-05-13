"""Streamlit app for multilingual academic article analysis."""

from __future__ import annotations

import base64
import html
import json
from pathlib import Path
from typing import Any

import streamlit as st
import streamlit.components.v1 as components

from config.settings import TEXT_PREVIEW_LIMIT
from modules.abstractive_engine import (
    summarize_english_transformer,
    summarize_turkish_hybrid_transformer,
    summarize_with_transformer,
)
from modules.evaluation import compare_all_summaries, compare_summaries
from modules.extractive_engine import summarize_with_textrank, summarize_with_tfidf
from modules.language_detector import detect_language
from modules.pdf_reader import PDFReadError, extract_text_from_pdf
from modules.preprocessing import preprocess_text


UI_TEXT = {
    "en": {
        "advanced_text_views": "Advanced text views",
        "all_methods_comparison_metrics": "All Methods Comparison Metrics",
        "app_subtitle": (
            "Compare TF-IDF, TextRank and Transformer-based summaries for Turkish "
            "and English academic PDFs."
        ),
        "app_title": "Multilingual Academic Article Summarizer",
        "bart_large_cnn": "BART Large CNN",
        "change_pdf": "Change PDF",
        "chunk_errors": "Chunk errors:",
        "chunk_summaries": "Chunk summaries",
        "chunks": "Chunks",
        "cleaned_characters": "Cleaned Characters",
        "cleaned_text": "Cleaned Text",
        "cleaned_text_preview": "Cleaned text preview",
        "cleaned_words": "Cleaned Words",
        "common_selected_sentences": "Common selected sentences",
        "comparison_metrics": "Comparison Metrics",
        "detected_language": "Detected Language",
        "debug_details": "Debug details",
        "document_statistics": "Document Statistics",
        "error": "Error",
        "extractive_ratio": "Extractive ratio",
        "file": "File",
        "generate_again": "Generate Again",
        "generate_summary": "Generate Summary",
        "hero_kicker": "Academic NLP Workspace",
        "input_words": "Input Words",
        "interface_language": "Interface language",
        "language_toggle": "Language",
        "jaccard_similarity": "Jaccard similarity",
        "language": "Language",
        "method": "Method",
        "method_compare_all": "Compare All",
        "method_compare_both": "Compare Both",
        "method_tfidf": "TF-IDF",
        "method_textrank": "TextRank",
        "method_transformer": "Transformer",
        "model": "Model",
        "model_id": "Model ID",
        "nlp_text": "NLP Text",
        "nlp_text_preview": "NLP text preview",
        "no_chunk_summaries": "No chunk summaries available.",
        "no_common_selected_sentences": "No common selected sentences.",
        "no_extractable_text": "No extractable text was found in this PDF.",
        "no_summary": "No summary could be generated from the current text.",
        "no_summary_generated": "No summary was generated.",
        "original": "Original",
        "original_text": "Original Text",
        "original_text_preview": "Original text preview",
        "preprocessing_debug": "Preprocessing debug",
        "preprocessing_failed": "Preprocessing failed to produce document statistics.",
        "please_upload_pdf": "Please upload a PDF to begin.",
        "processed_text": "Processed Text",
        "processed_text_preview": "Processed text preview",
        "project_intro": (
            "Upload a Turkish or English academic PDF and compare TF-IDF, "
            "TextRank, and Transformer summaries."
        ),
        "raw_characters": "Raw Characters",
        "raw_text": "Raw Text",
        "raw_text_preview": "Raw text preview",
        "raw_words": "Raw Words",
        "reading_pdf": "Reading PDF text...",
        "reduced_words": "Reduced Words",
        "reduction": "Reduction",
        "reduction_method": "Reduction method",
        "rejected_chunk_summaries": "Rejected chunk summaries:",
        "removed_metadata_lines": "Removed metadata lines",
        "removed_repeated_lines": "Removed repeated lines",
        "removed_short_noisy_lines": "Removed short/noisy lines",
        "removed_table_like_lines": "Removed table-like lines",
        "repaired_hyphenations": "Repaired hyphenations",
        "summary": "Summary",
        "summary_method": "Summarization method",
        "summary_ratio": "Summary ratio",
        "summary_results": "Summarization Results",
        "summary_words": "Summary Words",
        "summarization_text": "Summarization Text",
        "summarization_text_preview": "Summarization text preview",
        "summarization_words": "Summarization Words",
        "tfidf_compression": "TF-IDF compression",
        "tfidf_words": "TF-IDF words",
        "textrank_compression": "TextRank compression",
        "textrank_words": "TextRank words",
        "transformer_compression": "Transformer compression",
        "transformer_language_warning": (
            "Transformer summarization requires detected language to be English or Turkish."
        ),
        "transformer_loading": "Loading Transformer model and generating summary...",
        "transformer_summary": "Transformer Summary",
        "transformer_words": "Transformer words",
        "turkish_model": "Turkish model",
        "turkish_transformer_model": "Turkish Transformer model",
        "turkish_transformer_reduction_method": "Turkish Transformer reduction method",
        "upload_pdf_article": "Upload PDF article",
        "uploaded_file": "Uploaded file",
        "valid": "Valid",
        "warning": "Warning",
        "source_filtered_sentences": "Source-filtered sentences:",
        "detecting_language": "Detecting language...",
        "cleaning_text": "Cleaning academic text...",
        "generating_summary": "Generating summary...",
        "preparing_results": "Preparing results...",
        "ratio_10": "10% Short",
        "ratio_15": "15% Balanced",
        "ratio_20": "20% Detailed",
        "ratio_30": "30% Extended",
        "ratio_40": "40% Very detailed",
        "selected": "Selected",
        "selected_sentences": "selected sentences",
        "ratio": "Ratio",
        "original_words": "Original words",
    },
    "tr": {
        "advanced_text_views": "Gelişmiş metin görünümleri",
        "all_methods_comparison_metrics": "Tüm Yöntem Karşılaştırma Metrikleri",
        "app_subtitle": (
            "Türkçe ve İngilizce akademik PDF'ler için TF-IDF, TextRank ve "
            "Transformer tabanlı özetleri karşılaştırın."
        ),
        "app_title": "Çok Dilli Akademik Makale Özetleyici",
        "bart_large_cnn": "BART Large CNN",
        "change_pdf": "PDF'i değiştir",
        "chunk_errors": "Chunk hataları:",
        "chunk_summaries": "Chunk özetleri",
        "chunks": "Chunk",
        "cleaned_characters": "Temiz karakter",
        "cleaned_text": "Temizlenmiş Metin",
        "cleaned_text_preview": "Temizlenmiş metin önizleme",
        "cleaned_words": "Temiz kelime",
        "common_selected_sentences": "Ortak seçilen cümleler",
        "comparison_metrics": "Karşılaştırma Metrikleri",
        "detected_language": "Algılanan Dil",
        "debug_details": "Hata ayıklama detayları",
        "document_statistics": "Belge İstatistikleri",
        "error": "Hata",
        "extractive_ratio": "Extractive oran",
        "file": "Dosya",
        "generate_again": "Tekrar Özetle",
        "generate_summary": "Özet Oluştur",
        "hero_kicker": "Akademik NLP Çalışma Alanı",
        "input_words": "Girdi Kelime",
        "interface_language": "Arayüz dili",
        "language_toggle": "Dil",
        "jaccard_similarity": "Jaccard benzerliği",
        "language": "Dil",
        "method": "Yöntem",
        "method_compare_all": "Tümünü Karşılaştır",
        "method_compare_both": "İkisini Karşılaştır",
        "method_tfidf": "TF-IDF",
        "method_textrank": "TextRank",
        "method_transformer": "Transformer",
        "model": "Model",
        "model_id": "Model ID",
        "nlp_text": "NLP Metni",
        "nlp_text_preview": "NLP metni önizleme",
        "no_chunk_summaries": "Chunk özeti yok.",
        "no_common_selected_sentences": "Ortak seçilen cümle yok.",
        "no_extractable_text": "Bu PDF içinde çıkarılabilir metin bulunamadı.",
        "no_summary": "Mevcut metinden özet üretilemedi.",
        "no_summary_generated": "Özet oluşturulmadı.",
        "original": "Orijinal",
        "original_text": "Orijinal Metin",
        "original_text_preview": "Orijinal metin önizleme",
        "preprocessing_debug": "Ön işleme detayları",
        "preprocessing_failed": "Ön işleme belge istatistiklerini üretemedi.",
        "please_upload_pdf": "Başlamak için lütfen bir PDF yükleyin.",
        "processed_text": "İşlenmiş Metin",
        "processed_text_preview": "İşlenmiş metin önizleme",
        "project_intro": (
            "Türkçe veya İngilizce akademik PDF yükleyin; TF-IDF, TextRank "
            "ve Transformer özetlerini karşılaştırın."
        ),
        "raw_characters": "Ham karakter",
        "raw_text": "Ham Metin",
        "raw_text_preview": "Ham metin önizleme",
        "raw_words": "Ham kelime",
        "reading_pdf": "PDF metni okunuyor...",
        "reduced_words": "Azaltılmış Kelime",
        "reduction": "Azaltma",
        "reduction_method": "Azaltma yöntemi",
        "rejected_chunk_summaries": "Reddedilen chunk özetleri:",
        "removed_metadata_lines": "Kaldırılan metadata satırları",
        "removed_repeated_lines": "Kaldırılan tekrar satırları",
        "removed_short_noisy_lines": "Kaldırılan kısa/gürültülü satırlar",
        "removed_table_like_lines": "Kaldırılan tablo benzeri satırlar",
        "repaired_hyphenations": "Düzeltilen tire bölünmeleri",
        "summary": "Özet",
        "summary_method": "Özetleme yöntemi",
        "summary_ratio": "Özet oranı",
        "summary_results": "Özetleme Sonuçları",
        "summary_words": "Özet Kelime",
        "summarization_text": "Özetleme Metni",
        "summarization_text_preview": "Özetleme metni önizleme",
        "summarization_words": "Özetleme kelime",
        "tfidf_compression": "TF-IDF sıkıştırma",
        "tfidf_words": "TF-IDF kelime",
        "textrank_compression": "TextRank sıkıştırma",
        "textrank_words": "TextRank kelime",
        "transformer_compression": "Transformer sıkıştırma",
        "transformer_language_warning": (
            "Transformer özetleme için algılanan dil İngilizce veya Türkçe olmalıdır."
        ),
        "transformer_loading": "Transformer modeli yükleniyor ve özet oluşturuluyor...",
        "transformer_summary": "Transformer Özeti",
        "transformer_words": "Transformer kelime",
        "turkish_model": "Türkçe model",
        "turkish_transformer_model": "Türkçe Transformer modeli",
        "turkish_transformer_reduction_method": "Türkçe Transformer azaltma yöntemi",
        "upload_pdf_article": "PDF makale yükle",
        "uploaded_file": "Yüklenen dosya",
        "valid": "Geçerli",
        "warning": "Uyarı",
        "source_filtered_sentences": "Kaynak filtresinden çıkan cümleler:",
        "detecting_language": "Dil algılanıyor...",
        "cleaning_text": "Akademik metin temizleniyor...",
        "generating_summary": "Özet oluşturuluyor...",
        "preparing_results": "Sonuçlar hazırlanıyor...",
        "ratio_10": "10% Kısa",
        "ratio_15": "15% Dengeli",
        "ratio_20": "20% Detaylı",
        "ratio_30": "30% Geniş",
        "ratio_40": "40% Çok detaylı",
        "selected": "Seçilen",
        "selected_sentences": "seçilen cümleler",
        "ratio": "Oran",
        "original_words": "Orijinal kelime",
    },
}


METHOD_LABEL_KEYS = {
    "TF-IDF": "method_tfidf",
    "TextRank": "method_textrank",
    "Transformer": "method_transformer",
    "Compare Both": "method_compare_both",
    "Compare All": "method_compare_all",
}


RATIO_LABEL_KEYS = {
    0.10: "ratio_10",
    0.15: "ratio_15",
    0.20: "ratio_20",
    0.30: "ratio_30",
    0.40: "ratio_40",
}


UI_LANGUAGE_OPTIONS = {
    "English": "en",
    "Türkçe": "tr",
}


def get_ui_language() -> str:
    """Return the selected interface language without affecting NLP language detection."""
    query_language = str(st.query_params.get("ui_lang", ""))
    if query_language in UI_TEXT:
        st.session_state["ui_language"] = query_language
        return query_language

    language = str(st.session_state.get("ui_language", "en"))
    if language in UI_LANGUAGE_OPTIONS:
        language = UI_LANGUAGE_OPTIONS[language]
    return language if language in UI_TEXT else "en"


def t(key: str) -> str:
    """Translate a UI label with English fallback."""
    language = get_ui_language()
    return UI_TEXT.get(language, UI_TEXT["en"]).get(key, UI_TEXT["en"].get(key, key))


def format_method_label(method: str) -> str:
    """Return a localized label for a stable summarization method value."""
    return t(METHOD_LABEL_KEYS.get(method, method))


def format_ratio_label(value: float) -> str:
    """Return a localized label for a stable numeric summary ratio."""
    return t(RATIO_LABEL_KEYS.get(value, "summary_ratio"))


def render_language_toggle() -> str:
    """Render a plain text interface-language switch."""
    current_language = get_ui_language()
    next_language = "tr" if current_language == "en" else "en"
    label = "TR" if current_language == "en" else "EN"
    st.markdown(
        f"""
        <a class="lang-text-toggle" href="?ui_lang={next_language}" target="_self"
           title="{html.escape(t("language_toggle"))}">
            {label}
        </a>
        """,
        unsafe_allow_html=True,
    )
    st.session_state["ui_language"] = current_language
    return current_language


def _word_count(text: Any) -> int:
    """Count words for UI-only metrics."""
    return len(str(text or "").split())


def friendly_reduction_name(value: Any) -> str:
    """Return a compact display label for a Turkish reduction method."""
    return "TF-IDF" if str(value).lower() == "tfidf" else "TextRank"


def friendly_model_name(result: dict[str, Any]) -> str:
    """Return a compact display label for Transformer model metrics."""
    turkish_model = str(result.get("turkish_model", "")).lower()
    if turkish_model == "vbart_xlarge":
        return "VBART XLarge"
    if turkish_model == "mt5":
        return "mT5"

    model_name = str(result.get("model_name", ""))
    if model_name == "facebook/bart-large-cnn":
        return t("bart_large_cnn")
    if model_name == "vngrs-ai/VBART-XLarge-Summarization":
        return "VBART XLarge"
    if model_name == "mukayese/mt5-base-turkish-summarization":
        return "mT5"

    return model_name.rsplit("/", maxsplit=1)[-1] if "/" in model_name else model_name


def render_metric_grid(metrics: list[tuple[str, Any]], max_columns: int = 4) -> None:
    """Render compact metric cards in responsive rows."""
    clean_metrics = [(label, value) for label, value in metrics if value is not None]
    for start in range(0, len(clean_metrics), max_columns):
        row_metrics = clean_metrics[start : start + max_columns]
        columns = st.columns(len(row_metrics))
        for column, (label, value) in zip(columns, row_metrics):
            column.metric(label, value)


def load_local_image_as_base64(image_path: str) -> str:
    """Load a local image file and return it as a base64-encoded string."""
    resolved_path = Path(image_path)
    if not resolved_path.is_absolute():
        resolved_path = Path(__file__).resolve().parent / resolved_path

    try:
        return base64.b64encode(resolved_path.read_bytes()).decode("utf-8")
    except OSError:
        return ""


def load_local_text(image_path: str) -> str:
    """Load a local text asset and return an empty string if unavailable."""
    resolved_path = Path(image_path)
    if not resolved_path.is_absolute():
        resolved_path = Path(__file__).resolve().parent / resolved_path

    try:
        return resolved_path.read_text(encoding="utf-8")
    except OSError:
        return ""


def inject_custom_css(background_path: str = "assets/summarizer-bg.jpeg") -> None:
    """Inject the custom dark glassmorphism theme for the Streamlit app."""
    background_base64 = load_local_image_as_base64(background_path)
    if background_base64:
        background_image = f"url('data:image/jpeg;base64,{background_base64}')"
    else:
        background_image = (
            "radial-gradient(circle at 20% 12%, rgba(57, 217, 255, 0.34), transparent 32%), "
            "radial-gradient(circle at 82% 22%, rgba(59, 130, 246, 0.24), transparent 34%), "
            "linear-gradient(135deg, #020617 0%, #071426 48%, #020617 100%)"
        )

    st.markdown(
        f"""
        <style>
        :root {{
            --app-bg: #050b16;
            --glass-bg: rgba(3, 7, 18, 0.62);
            --glass-bg-strong: rgba(3, 7, 18, 0.74);
            --glass-border: rgba(125, 211, 252, 0.18);
            --text-main: #edf7ff;
            --text-muted: #b7c7d9;
            --accent-cyan: #39d9ff;
            --accent-blue: #3b82f6;
            --accent-deep: #0f2f5f;
        }}

        html, body, .stApp {{
            font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }}

        .stApp {{
            background-image: {background_image};
            background-size: cover;
            background-position: center top;
            background-attachment: fixed;
            color: var(--text-main);
        }}

        [data-testid="stToolbar"],
        [data-testid="stDecoration"],
        [data-testid="stStatusWidget"],
        #MainMenu,
        footer,
        header {{
            display: none !important;
            visibility: hidden !important;
            height: 0 !important;
        }}

        .stApp::before {{
            content: "";
            position: fixed;
            inset: 0;
            z-index: 0;
            pointer-events: none;
            background: rgba(2, 6, 23, 0.35);
        }}

        .block-container {{
            position: relative;
            z-index: 1;
            max-width: min(1500px, 95vw);
            padding-top: 0.75rem;
            padding-left: 1.5rem;
            padding-right: 1.5rem;
            padding-bottom: 4rem;
            animation: fadeIn 380ms ease both;
        }}

        [data-testid="stSidebar"] {{
            background: rgba(3, 7, 18, 0.72);
            border-right: 1px solid rgba(125, 211, 252, 0.14);
            backdrop-filter: blur(10px);
        }}

        [data-testid="stSidebar"] > div {{
            background: transparent;
            padding-top: 1.4rem;
        }}

        .center-shell {{
            max-width: 920px;
            margin: 1.4rem auto 0;
            animation: slideUp 420ms ease both;
        }}

        .control-card,
        .analysis-card {{
            padding: 1rem 1.1rem;
            border: 1px solid var(--glass-border);
            border-radius: 22px;
            background:
                linear-gradient(145deg, rgba(3, 7, 18, 0.74), rgba(5, 13, 28, 0.56)),
                var(--glass-bg);
            box-shadow: 0 18px 54px rgba(0, 0, 0, 0.28);
            backdrop-filter: blur(10px);
        }}

        .control-card {{
            padding: 1.35rem;
            margin-bottom: 1rem;
            animation: slideUp 420ms ease both;
        }}

        .analysis-card {{
            position: sticky;
            top: 1rem;
            padding: 0.85rem 0.95rem;
            animation: slideInLeft 420ms ease both;
        }}

        .control-card h1 {{
            margin: 0;
            color: #f7fbff;
            font-size: clamp(1.55rem, 3vw, 2.35rem);
            line-height: 1.12;
            letter-spacing: 0;
        }}

        .control-card p {{
            max-width: 760px;
            margin: 0.55rem 0 0;
            color: var(--text-muted);
            font-size: 0.94rem;
            line-height: 1.55;
        }}

        .hero-section {{
            padding: 0.55rem 0.8rem;
            margin-bottom: 0.8rem;
            border: 1px solid var(--glass-border);
            border-radius: 16px;
            background:
                linear-gradient(135deg, rgba(3, 7, 18, 0.72), rgba(7, 16, 34, 0.52)),
                linear-gradient(90deg, rgba(57, 217, 255, 0.10), rgba(59, 130, 246, 0.05));
            box-shadow: 0 12px 34px rgba(0, 0, 0, 0.24);
            backdrop-filter: blur(10px);
            animation: slideUp 360ms ease both;
        }}

        .hero-section .hero-kicker {{
            margin: 0 0 0.3rem;
            color: var(--accent-cyan);
            font-size: 0.62rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }}

        .hero-section h1 {{
            margin: 0;
            color: #f8fcff;
            font-size: clamp(1.05rem, 1.7vw, 1.45rem);
            line-height: 1.1;
            letter-spacing: 0;
        }}

        .hero-section p {{
            max-width: 820px;
            margin: 0.28rem 0 0;
            color: var(--text-muted);
            font-size: 0.82rem;
            line-height: 1.45;
        }}

        .header-copy {{
            padding: 0.15rem 0 0.25rem;
        }}

        .header-copy .hero-kicker {{
            margin: 0 0 0.3rem;
            color: var(--accent-cyan);
            font-size: 0.62rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }}

        .header-copy h1 {{
            margin: 0;
            color: #f8fcff;
            font-size: clamp(1.55rem, 3vw, 2.35rem);
            line-height: 1.12;
            letter-spacing: 0;
        }}

        .header-copy.compact h1 {{
            font-size: clamp(1.05rem, 1.7vw, 1.45rem);
            line-height: 1.1;
        }}

        .header-copy p {{
            max-width: 820px;
            margin: 0.55rem 0 0;
            color: var(--text-muted);
            font-size: 0.94rem;
            line-height: 1.55;
        }}

        .header-copy.compact p {{
            margin-top: 0.28rem;
            font-size: 0.82rem;
            line-height: 1.45;
        }}

        .language-toggle-slot {{
            display: flex;
            justify-content: flex-end;
            align-items: flex-start;
            padding-top: 0;
            margin-top: -0.25rem;
        }}

        .lang-text-toggle {{
            display: inline-block;
            color: rgba(226, 238, 249, 0.62) !important;
            font-size: 0.68rem;
            font-weight: 700;
            line-height: 1;
            letter-spacing: 0.04em;
            text-decoration: none !important;
            background: transparent !important;
            border: 0 !important;
            box-shadow: none !important;
            padding: 0;
        }}

        .lang-text-toggle:hover {{
            color: rgba(247, 252, 255, 0.95) !important;
            text-decoration: underline !important;
            text-underline-offset: 3px;
        }}

        .summary-card {{
            margin: 0.5rem 0 1rem;
            width: 100%;
            padding: 1.1rem 1.2rem;
            border: 1px solid rgba(125, 211, 252, 0.16);
            border-radius: 18px;
            background: rgba(2, 6, 15, 0.84);
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.04), 0 14px 32px rgba(0, 0, 0, 0.22);
            backdrop-filter: blur(8px);
            animation: slideUp 360ms ease both;
        }}

        .summary-card p {{
            margin: 0;
            color: #edf7ff;
            font-size: 16px;
            line-height: 1.78;
        }}

        .uploaded-file-card {{
            margin-bottom: 1rem;
            padding: 0.7rem 0.85rem;
            border: 1px solid rgba(125, 211, 252, 0.14);
            border-radius: 14px;
            background: rgba(3, 7, 18, 0.62);
            color: var(--text-muted);
            font-size: 0.9rem;
            animation: fadeIn 360ms ease both;
        }}

        .side-stat {{
            padding: 0.65rem 0;
            border-bottom: 1px solid rgba(125, 211, 252, 0.11);
        }}

        .side-stat:last-child {{
            border-bottom: 0;
        }}

        .side-stat span {{
            display: block;
            color: #9fb5c9;
            font-size: 0.73rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }}

        .side-stat strong {{
            display: block;
            margin-top: 0.15rem;
            color: #f3fbff;
            font-size: 0.9rem;
            line-height: 1.35;
            word-break: break-word;
        }}

        h1, h2, h3, h4, h5, h6,
        [data-testid="stMarkdownContainer"] h1,
        [data-testid="stMarkdownContainer"] h2,
        [data-testid="stMarkdownContainer"] h3 {{
            color: #f7fbff;
            letter-spacing: 0;
        }}

        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] li,
        [data-testid="stMarkdownContainer"] div {{
            color: var(--text-main);
        }}

        [data-testid="stFileUploader"],
        [data-testid="stTextArea"],
        [data-testid="stTable"],
        [data-testid="stAlert"],
        [data-testid="stExpander"],
        [data-testid="stVerticalBlockBorderWrapper"] {{
            border-radius: 18px;
        }}

        [data-testid="stTextArea"] {{
            max-width: 1080px;
        }}

        [data-testid="stVerticalBlockBorderWrapper"] {{
            border: 1px solid var(--glass-border);
            background: rgba(3, 7, 18, 0.62);
            box-shadow: 0 16px 44px rgba(0, 0, 0, 0.24);
            backdrop-filter: blur(10px);
            animation: slideUp 380ms ease both;
        }}

        [data-testid="stFileUploader"] section {{
            border: 1px dashed rgba(57, 217, 255, 0.46);
            border-radius: 18px;
            background: rgba(3, 7, 18, 0.62);
            backdrop-filter: blur(8px);
        }}

        [data-testid="stFileUploader"] small,
        [data-testid="stFileUploader"] span {{
            color: var(--text-muted);
        }}

        div[data-testid="stMetric"] {{
            min-height: 96px;
            padding: 0.82rem 0.9rem;
            border: 1px solid rgba(125, 211, 252, 0.17);
            border-radius: 18px;
            background: linear-gradient(145deg, rgba(3, 7, 18, 0.76), rgba(7, 15, 30, 0.56));
            box-shadow: 0 12px 30px rgba(0, 0, 0, 0.20);
            backdrop-filter: blur(10px);
            animation: fadeIn 420ms ease both;
        }}

        div[data-testid="stMetric"] label,
        div[data-testid="stMetric"] [data-testid="stMetricLabel"] {{
            color: #aac1d8;
            font-weight: 650;
        }}

        div[data-testid="stMetric"] [data-testid="stMetricValue"] {{
            color: #f4fbff;
            font-size: 1.22rem;
            word-break: break-word;
        }}

        .stButton button {{
            min-height: 3rem;
            border: 0;
            border-radius: 999px;
            padding: 0.72rem 1.3rem;
            color: #03101d;
            font-weight: 800;
            background: linear-gradient(135deg, var(--accent-cyan), var(--accent-blue));
            box-shadow: 0 12px 28px rgba(57, 217, 255, 0.20);
            transition: transform 180ms ease, box-shadow 180ms ease, filter 180ms ease;
        }}

        .stButton button:hover {{
            color: #020814;
            filter: brightness(1.07);
            transform: translateY(-1px);
            box-shadow: 0 16px 38px rgba(57, 217, 255, 0.30);
        }}

        .stButton button:focus {{
            color: #020814;
            box-shadow: 0 0 0 3px rgba(57, 217, 255, 0.25), 0 18px 40px rgba(57, 217, 255, 0.28);
        }}

        .stButton button:disabled {{
            color: rgba(237, 247, 255, 0.46);
            background: rgba(68, 83, 104, 0.42);
            box-shadow: none;
            transform: none;
        }}

        [data-testid="stSelectbox"] div[role="combobox"] {{
            border: 1px solid rgba(125, 211, 252, 0.22);
            border-radius: 14px;
            background: rgba(3, 7, 18, 0.72);
            color: var(--text-main);
            box-shadow: none;
        }}

        [data-testid="stTabs"] {{
            padding: 0.75rem;
            border: 1px solid rgba(125, 211, 252, 0.16);
            border-radius: 20px;
            background: rgba(3, 7, 18, 0.62);
            backdrop-filter: blur(10px);
            animation: slideUp 420ms ease both;
        }}

        [data-testid="stTabs"] div[role="tablist"] {{
            gap: 0.45rem;
            padding: 0.28rem;
            border: 1px solid rgba(125, 211, 252, 0.12);
            border-radius: 14px;
            background: rgba(1, 5, 14, 0.30);
        }}

        [data-testid="stTabs"] button[role="tab"] {{
            height: 2.55rem;
            padding: 0 1rem;
            border: 1px solid transparent;
            border-radius: 12px;
            color: #adc2d7;
            background: transparent;
            transition: background 160ms ease, border-color 160ms ease, color 160ms ease;
        }}

        [data-testid="stTabs"] button[role="tab"][aria-selected="true"] {{
            color: #f8fcff;
            border-color: rgba(57, 217, 255, 0.36);
            background: rgba(57, 217, 255, 0.13);
            box-shadow: inset 0 -2px 0 rgba(57, 217, 255, 0.42);
        }}

        [data-testid="stExpander"] {{
            border: 1px solid rgba(125, 211, 252, 0.15);
            background: rgba(3, 7, 18, 0.62);
            backdrop-filter: blur(10px);
            overflow: hidden;
            animation: fadeIn 360ms ease both;
        }}

        [data-testid="stExpander"] summary {{
            color: #ecf8ff;
            font-weight: 700;
        }}

        [data-testid="stTextArea"] textarea {{
            max-width: 1080px;
            border: 1px solid rgba(125, 211, 252, 0.17) !important;
            border-radius: 16px !important;
            background: rgba(1, 5, 14, 0.88) !important;
            color: #ecf8ff !important;
            line-height: 1.62 !important;
        }}

        [data-testid="stTable"] {{
            width: 100%;
        }}

        [data-testid="stTable"] table {{
            width: 100%;
            color: var(--text-main);
            background: rgba(3, 7, 18, 0.72);
        }}

        [data-testid="stTable"] th {{
            color: #f6fbff;
            background: rgba(57, 217, 255, 0.12);
        }}

        [data-testid="stTable"] td {{
            color: var(--text-main);
            background: rgba(3, 10, 22, 0.54);
        }}

        hr {{
            border-color: rgba(148, 224, 255, 0.14);
        }}

        [data-testid="stProgress"] {{
            margin-bottom: 0.25rem;
        }}

        [data-testid="stProgress"] > div {{
            height: 0.42rem;
        }}

        @keyframes fadeIn {{
            from {{
                opacity: 0;
            }}
            to {{
                opacity: 1;
            }}
        }}

        @keyframes slideUp {{
            from {{
                opacity: 0;
                transform: translateY(12px);
            }}
            to {{
                opacity: 1;
                transform: translateY(0);
            }}
        }}

        @keyframes slideInLeft {{
            from {{
                opacity: 0;
                transform: translateX(-14px);
            }}
            to {{
                opacity: 1;
                transform: translateX(0);
            }}
        }}

        @keyframes slideInRight {{
            from {{
                opacity: 0;
                transform: translateX(14px);
            }}
            to {{
                opacity: 1;
                transform: translateX(0);
            }}
        }}

        @media (max-width: 768px) {{
            .block-container {{
                padding-top: 1.4rem;
                padding-left: 1rem;
                padding-right: 1rem;
            }}

            .hero-section {{
                padding: 0.85rem;
                border-radius: 18px;
            }}

            .hero-section p {{
                font-size: 0.86rem;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    """Render the application hero section."""
    with st.container(border=True):
        text_col, toggle_col = st.columns([0.96, 0.04])
        with toggle_col:
            st.markdown('<div class="language-toggle-slot">', unsafe_allow_html=True)
            render_language_toggle()
            st.markdown("</div>", unsafe_allow_html=True)
        with text_col:
            st.markdown(
                f"""
                <div class="header-copy compact">
                    <p class="hero-kicker">{html.escape(t("hero_kicker"))}</p>
                    <h1>{html.escape(t("app_title"))}</h1>
                    <p>{html.escape(t("app_subtitle"))}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_center_header() -> None:
    """Render the compact centered workspace introduction."""
    with st.container(border=True):
        text_col, toggle_col = st.columns([0.96, 0.04])
        with toggle_col:
            st.markdown('<div class="language-toggle-slot">', unsafe_allow_html=True)
            render_language_toggle()
            st.markdown("</div>", unsafe_allow_html=True)
        with text_col:
            st.markdown(
                f"""
                <div class="header-copy">
                    <h1>{html.escape(t("app_title"))}</h1>
                    <p>{html.escape(t("project_intro"))}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_text_card(text: str) -> None:
    """Render long generated text inside a readable glass card."""
    safe_text = html.escape(text).replace("\n", "<br>")
    st.markdown(
        f"""
        <div class="summary-card">
            <p>{safe_text}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_loading_status(
    message: str,
    animation_path: str = "assets/loading-animation.json",
) -> None:
    """Render one compact inline loading status with a local Lottie fallback."""
    animation_json = load_local_text(animation_path)
    try:
        animation_data = json.dumps(json.loads(animation_json))
    except json.JSONDecodeError:
        animation_data = "{}"

    components.html(
        f"""
        <style>
            html, body {{
                margin: 0;
                padding: 0;
                background: transparent;
                overflow: hidden;
            }}

            .loading-inline {{
                display: inline-flex;
                align-items: center;
                justify-content: flex-start;
                gap: 0.55rem;
                width: 100%;
                height: 44px;
                max-height: 44px;
                overflow: hidden;
                color: #edf7ff;
                font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
                font-size: 15px;
                font-weight: 700;
            }}

            .loading-icon {{
                display: inline-flex;
                align-items: center;
                justify-content: center;
                width: 36px;
                height: 36px;
                max-width: 36px;
                max-height: 36px;
                overflow: hidden;
                flex: 0 0 36px;
            }}

            .loading-icon svg {{
                width: auto !important;
                max-width: 36px !important;
                height: auto !important;
                max-height: 36px !important;
                display: block;
            }}

            .loading-fallback {{
                color: #39d9ff;
                font-size: 22px;
                line-height: 1;
                animation: pulse 900ms ease-in-out infinite alternate;
            }}

            @keyframes pulse {{
                from {{
                    opacity: 0.58;
                    transform: scale(0.96);
                }}
                to {{
                    opacity: 1;
                    transform: scale(1.04);
                }}
            }}
        </style>
        <div class="loading-inline">
            <div id="loading-icon" class="loading-icon">
                <span class="loading-fallback">✨</span>
            </div>
            <span>{html.escape(message)}</span>
        </div>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/lottie-web/5.12.2/lottie.min.js"></script>
        <script>
            const container = document.getElementById("loading-icon");
            if (window.lottie && container) {{
                container.innerHTML = "";
                window.lottie.loadAnimation({{
                    container,
                    renderer: "svg",
                    loop: true,
                    autoplay: true,
                    rendererSettings: {{
                        preserveAspectRatio: "xMidYMid meet"
                    }},
                    animationData: {animation_data}
                }});
            }}
        </script>
        """,
        height=44,
    )


def update_loading_status(status_slot: Any, message: str) -> None:
    """Update the single inline loading status placeholder."""
    status_slot.empty()
    with status_slot.container():
        render_loading_status(message)


def render_summary_controls(
    button_label: str,
    *,
    disabled: bool = False,
    detected_language: str | None = None,
) -> tuple[str, float, str, str, bool]:
    """Render method, ratio, and action controls."""
    summary_method = st.selectbox(
        t("summary_method"),
        options=["TF-IDF", "TextRank", "Transformer", "Compare Both", "Compare All"],
        format_func=format_method_label,
        key="summary_method",
    )
    summary_ratio = st.selectbox(
        t("summary_ratio"),
        options=list(RATIO_LABEL_KEYS.keys()),
        index=1,
        format_func=format_ratio_label,
        key="summary_ratio",
    )
    turkish_model_key = render_turkish_model_control(
        detected_language,
        str(summary_method),
    )
    turkish_reduction_method = render_turkish_reduction_method_control(
        detected_language,
        str(summary_method),
    )
    generate_clicked = st.button(
        button_label,
        disabled=disabled,
        use_container_width=True,
    )
    return (
        str(summary_method),
        float(summary_ratio),
        turkish_reduction_method,
        turkish_model_key,
        generate_clicked,
    )


def _should_show_turkish_transformer_options(
    detected_language: str | None,
    summary_method: str,
) -> bool:
    """Return whether Turkish Transformer controls should be visible."""
    normalized_language = (detected_language or "").lower().strip()
    return (
        normalized_language == "tr"
        and summary_method in {"Transformer", "Compare All"}
    )


def render_turkish_reduction_method_control(
    detected_language: str | None,
    summary_method: str,
) -> str:
    """Render Turkish-only Transformer reduction controls when enough context exists."""
    current_value = str(st.session_state.get("turkish_reduction_method", "textrank"))
    if not _should_show_turkish_transformer_options(detected_language, summary_method):
        return current_value

    return str(
        st.selectbox(
            t("turkish_transformer_reduction_method"),
            options=["textrank", "tfidf"],
            index=0 if current_value != "tfidf" else 1,
            format_func=lambda value: "TextRank" if value == "textrank" else "TF-IDF",
            key="turkish_reduction_method",
        )
    )


def render_turkish_model_control(
    detected_language: str | None,
    summary_method: str,
) -> str:
    """Render Turkish-only Transformer model selection when available."""
    current_value = str(st.session_state.get("turkish_model_key", "mt5"))
    if not _should_show_turkish_transformer_options(detected_language, summary_method):
        return current_value

    return str(
        st.selectbox(
            t("turkish_transformer_model"),
            options=["mt5", "vbart_xlarge"],
            index=0 if current_value != "vbart_xlarge" else 1,
            format_func=lambda value: "mT5" if value == "mt5" else "VBART XLarge",
            key="turkish_model_key",
        )
    )


def render_method_controls(
    button_label: str,
    *,
    disabled: bool = False,
    detected_language: str | None = None,
) -> tuple[Any, str, float, str, str, bool]:
    """Render upload and summarization controls and return their values."""
    uploaded_file = st.file_uploader(t("upload_pdf_article"), type=["pdf"], key="pdf_upload")
    if uploaded_file is None:
        st.caption(t("please_upload_pdf"))
    (
        summary_method,
        summary_ratio,
        turkish_reduction_method,
        turkish_model_key,
        generate_clicked,
    ) = render_summary_controls(
        button_label,
        disabled=disabled,
        detected_language=detected_language,
    )
    return (
        uploaded_file,
        str(summary_method),
        float(summary_ratio),
        turkish_reduction_method,
        turkish_model_key,
        generate_clicked,
    )


def render_uploaded_file_card(file_name: str) -> None:
    """Render the uploaded file name in a compact glass panel."""
    st.markdown(
        f"""
        <div class="uploaded-file-card">
            <strong>{html.escape(t("uploaded_file"))}:</strong> {html.escape(file_name)}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_document_statistics(detected_language: str, stats: dict[str, Any]) -> None:
    """Render document statistics as compact metric cards."""
    st.subheader(t("document_statistics"))
    render_metric_grid(
        [
            (t("language"), detected_language),
            (t("raw_words"), stats["raw_word_count"]),
            (t("cleaned_words"), stats["cleaned_word_count"]),
            (t("summarization_words"), stats.get("summarization_word_count", "—")),
            (t("raw_characters"), stats["raw_character_count"]),
            (t("cleaned_characters"), stats["cleaned_character_count"]),
        ],
    )


def render_preprocessing_debug(stats: dict[str, Any]) -> None:
    """Render optional preprocessing removal counts without crowding metrics."""
    with st.expander(t("preprocessing_debug")):
        st.write(f"{t('removed_repeated_lines')}: {stats.get('removed_repeated_line_count', 0)}")
        st.write(f"{t('removed_metadata_lines')}: {stats.get('removed_metadata_line_count', 0)}")
        st.write(f"{t('removed_table_like_lines')}: {stats.get('removed_table_line_count', 0)}")
        st.write(f"{t('removed_short_noisy_lines')}: {stats.get('removed_short_noise_line_count', 0)}")
        st.write(f"{t('repaired_hyphenations')}: {stats.get('repaired_hyphenation_count', 0)}")


def render_text_tabs(
    extracted_text: str,
    display_text: str,
    nlp_text: str,
    summarization_text: str = "",
) -> None:
    """Render compact text previews with advanced debug views available."""
    processed_text = summarization_text or display_text
    original_tab, processed_tab = st.tabs([t("original_text"), t("processed_text")])

    with original_tab:
        st.text_area(
            t("original_text_preview"),
            extracted_text[:TEXT_PREVIEW_LIMIT],
            height=350,
        )

    with processed_tab:
        st.text_area(
            t("processed_text_preview"),
            processed_text[:TEXT_PREVIEW_LIMIT],
            height=350,
        )

    with st.expander(t("advanced_text_views")):
        raw_tab, cleaned_tab, summ_tab, nlp_tab = st.tabs(
            [t("raw_text"), t("cleaned_text"), t("summarization_text"), t("nlp_text")]
        )

        with raw_tab:
            st.text_area(
                t("raw_text_preview"),
                extracted_text[:TEXT_PREVIEW_LIMIT],
                height=280,
            )

        with cleaned_tab:
            st.text_area(
                t("cleaned_text_preview"),
                display_text[:TEXT_PREVIEW_LIMIT],
                height=280,
            )

        with summ_tab:
            st.text_area(
                t("summarization_text_preview"),
                processed_text[:TEXT_PREVIEW_LIMIT],
                height=280,
            )

        with nlp_tab:
            st.text_area(
                t("nlp_text_preview"),
                nlp_text[:TEXT_PREVIEW_LIMIT],
                height=280,
            )


def render_analysis_control_card(
    file_name: str,
    detected_language: str,
    stats: dict[str, Any],
    summary_method: str,
    summary_ratio: float,
) -> None:
    """Render the left-side analysis control and document summary card."""
    st.markdown(
        f"""
        <div class="analysis-card">
            <div class="side-stat">
                <span>{html.escape(t("file"))}</span>
                <strong>{html.escape(file_name)}</strong>
            </div>
            <div class="side-stat">
                <span>{html.escape(t("detected_language"))}</span>
                <strong>{html.escape(detected_language)}</strong>
            </div>
            <div class="side-stat">
                <span>{html.escape(t("method"))}</span>
                <strong>{html.escape(format_method_label(summary_method))}</strong>
            </div>
            <div class="side-stat">
                <span>{html.escape(t("ratio"))}</span>
                <strong>{summary_ratio:.0%}</strong>
            </div>
            <div class="side-stat">
                <span>{html.escape(t("raw_words"))}</span>
                <strong>{stats["raw_word_count"]}</strong>
            </div>
            <div class="side-stat">
                <span>{html.escape(t("cleaned_words"))}</span>
                <strong>{stats["cleaned_word_count"]}</strong>
            </div>
            <div class="side-stat">
                <span>{html.escape(t("summarization_words"))}</span>
                <strong>{stats.get("summarization_word_count", "—")}</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_summary_result(result: dict[str, Any]) -> None:
    """Render one summarization result in the Streamlit UI."""
    method = str(result["method"])

    st.markdown(f"### {method} {t('summary')}")
    if result["summary"]:
        render_text_card(str(result["summary"]))
    else:
        st.warning(t("no_summary"))

    if result.get("message"):
        st.info(str(result["message"]))

    render_metric_grid(
        [
            (t("language"), result.get("language")),
            (t("input_words"), result.get("input_word_count")),
            (t("summary_words"), result.get("summary_word_count")),
            (t("original"), result["original_sentence_count"]),
            (t("valid"), result["valid_sentence_count"]),
            (t("selected"), result["selected_sentence_count"]),
            (t("ratio"), f"{result['summary_ratio']:.0%}"),
        ],
    )

    with st.expander(f"{method} {t('selected_sentences')}"):
        for index, sentence in enumerate(result["selected_sentences"], start=1):
            st.write(f"{index}. {sentence}")


def render_comparison_metrics(comparison: dict[str, Any]) -> None:
    """Render TF-IDF and TextRank comparison metrics."""
    sentence_overlap = comparison["sentence_overlap"]
    st.markdown(f"### {t('comparison_metrics')}")
    render_metric_grid(
        [
            (t("original_words"), comparison["original_word_count"]),
            (t("tfidf_words"), comparison["tfidf_word_count"]),
            (t("textrank_words"), comparison["textrank_word_count"]),
            (t("tfidf_compression"), f"{comparison['tfidf_compression_ratio']:.2%}"),
            (t("textrank_compression"), f"{comparison['textrank_compression_ratio']:.2%}"),
            (t("common_selected_sentences"), sentence_overlap["common_sentence_count"]),
            (t("jaccard_similarity"), f"{sentence_overlap['jaccard_similarity']:.2%}"),
        ],
    )

    with st.expander(t("common_selected_sentences")):
        common_sentences = sentence_overlap["common_sentences"]
        if common_sentences:
            for index, sentence in enumerate(common_sentences, start=1):
                st.write(f"{index}. {sentence}")
        else:
            st.write(t("no_common_selected_sentences"))


def render_compare_both_results(
    tfidf_result: dict[str, Any],
    textrank_result: dict[str, Any],
    original_text: str,
) -> None:
    """Render both extractive summaries and their shared comparison metrics."""
    tfidf_tab, textrank_tab = st.tabs(["TF-IDF", "TextRank"])

    with tfidf_tab:
        render_summary_result(tfidf_result)

    with textrank_tab:
        render_summary_result(textrank_result)

    st.divider()
    comparison_container = st.container()
    with comparison_container:
        render_comparison_metrics(
            compare_summaries(tfidf_result, textrank_result, original_text)
        )


def render_all_methods_comparison_metrics(comparison: dict[str, Any]) -> None:
    """Render comparison metrics for all available summarization methods."""
    sentence_overlap = comparison["sentence_overlap"]
    st.markdown(f"### {t('all_methods_comparison_metrics')}")
    render_metric_grid(
        [
            (t("original_words"), comparison["original_word_count"]),
            (t("tfidf_words"), comparison["tfidf_word_count"]),
            (t("textrank_words"), comparison["textrank_word_count"]),
            (t("transformer_words"), comparison["transformer_word_count"]),
            (t("tfidf_compression"), f"{comparison['tfidf_compression_ratio']:.2%}"),
            (t("textrank_compression"), f"{comparison['textrank_compression_ratio']:.2%}"),
            (t("transformer_compression"), f"{comparison['transformer_compression_ratio']:.2%}"),
            (t("common_selected_sentences"), sentence_overlap["common_sentence_count"]),
            (t("jaccard_similarity"), f"{sentence_overlap['jaccard_similarity']:.2%}"),
        ],
    )


def render_compare_all_results(
    tfidf_result: dict[str, Any],
    textrank_result: dict[str, Any],
    transformer_result: dict[str, Any],
    original_text: str,
) -> None:
    """Render all summarization methods and shared comparison metrics."""
    tfidf_tab, textrank_tab, transformer_tab = st.tabs(
        ["TF-IDF", "TextRank", "Transformer"]
    )

    with tfidf_tab:
        render_summary_result(tfidf_result)

    with textrank_tab:
        render_summary_result(textrank_result)

    with transformer_tab:
        render_transformer_result(transformer_result)

    st.divider()
    render_all_methods_comparison_metrics(
        compare_all_summaries(
            tfidf_result,
            textrank_result,
            transformer_result,
            original_text,
        )
    )


def render_transformer_result(result: dict[str, Any]) -> None:
    """Render one Transformer summarization result in the Streamlit UI."""
    st.markdown(f"### {t('transformer_summary')}")

    error_message = result.get("error")
    warning_message = result.get("warning")
    chunk_errors = result.get("chunk_errors", [])
    rejected_chunk_summaries = result.get("rejected_chunk_summaries", [])
    source_filtered_sentences = result.get("source_filtered_sentences", [])

    if error_message:
        st.error(str(error_message))
    if warning_message:
        st.warning(str(warning_message))

    if result["summary"]:
        render_text_card(str(result["summary"]))
    elif not error_message:
        st.warning(t("no_summary_generated"))

    metrics = [
        (t("language"), result.get("language")),
        (t("model"), friendly_model_name(result)),
        (t("chunks"), result.get("chunk_count")),
        (t("input_words"), result.get("input_word_count")),
        (t("summary_words"), result.get("summary_word_count")),
    ]
    if "reduced_input_word_count" in result:
        metrics.append((t("reduced_words"), result["reduced_input_word_count"]))
    if "extractive_ratio" in result:
        metrics.append((t("ratio"), f"{result['extractive_ratio']:.0%}"))
    if "reduction_method" in result:
        metrics.append((t("reduction"), friendly_reduction_name(result["reduction_method"])))
    render_metric_grid(metrics)

    with st.expander(t("chunk_summaries")):
        chunk_summaries = result["chunk_summaries"]
        if chunk_summaries:
            for index, chunk_summary in enumerate(chunk_summaries, start=1):
                st.write(f"{index}. {chunk_summary}")
        else:
            st.write(t("no_chunk_summaries"))

    if (
        error_message
        or warning_message
        or rejected_chunk_summaries
        or source_filtered_sentences
    ):
        with st.expander(t("debug_details")):
            if error_message:
                st.write(f"{t('error')}: {error_message}")
            if warning_message:
                st.write(f"{t('warning')}: {warning_message}")
            st.write(f"{t('model_id')}: {result['model_name']}")
            if "turkish_model" in result:
                st.write(f"{t('turkish_model')}: {result['turkish_model']}")
            st.write(f"{t('detected_language')}: {result['language']}")
            st.write(f"{t('chunks')}: {result['chunk_count']}")
            st.write(f"{t('input_words')}: {result['input_word_count']}")
            if "reduced_input_word_count" in result:
                st.write(f"{t('reduced_words')}: {result['reduced_input_word_count']}")
            if "extractive_ratio" in result:
                st.write(f"{t('extractive_ratio')}: {result['extractive_ratio']:.2f}")
            if "reduction_method" in result:
                st.write(f"{t('reduction_method')}: {result['reduction_method']}")
            st.write(f"{t('summary_words')}: {result['summary_word_count']}")
            if chunk_errors:
                st.write(t("chunk_errors"))
                for error in chunk_errors:
                    st.write(f"- {error}")
            if rejected_chunk_summaries:
                st.write(t("rejected_chunk_summaries"))
                for index, summary in enumerate(rejected_chunk_summaries, start=1):
                    st.write(f"{index}. {summary}")
            if source_filtered_sentences:
                st.write(t("source_filtered_sentences"))
                for index, sentence in enumerate(source_filtered_sentences, start=1):
                    st.write(f"{index}. {sentence}")


def process_uploaded_pdf(
    uploaded_file: Any,
    progress_bar: Any | None = None,
    status_text: Any | None = None,
) -> dict[str, Any] | None:
    """Extract, detect, and preprocess one uploaded PDF file."""
    if progress_bar is not None:
        progress_bar.progress(5)
    if status_text is not None:
        update_loading_status(status_text, t("reading_pdf"))

    try:
        extracted_text = extract_text_from_pdf(uploaded_file.getvalue())
    except PDFReadError as error:
        st.error(str(error))
        return None

    if not extracted_text.strip():
        st.warning(t("no_extractable_text"))
        return None

    if progress_bar is not None:
        progress_bar.progress(25)
    if status_text is not None:
        update_loading_status(status_text, t("detecting_language"))
    detected_language = detect_language(extracted_text)

    if progress_bar is not None:
        progress_bar.progress(45)
    if status_text is not None:
        update_loading_status(status_text, t("cleaning_text"))
    preprocessing_result = preprocess_text(extracted_text, detected_language)
    stats = preprocessing_result["stats"]

    if not isinstance(stats, dict):
        st.error(t("preprocessing_failed"))
        return None

    summarization_text = str(preprocessing_result.get("summarization_text", ""))
    display_text = str(preprocessing_result["display_text"])

    return {
        "file_name": uploaded_file.name,
        "extracted_text": extracted_text,
        "detected_language": detected_language,
        "display_text": display_text,
        "summarization_text": summarization_text,
        "nlp_text": str(preprocessing_result["nlp_text"]),
        "stats": stats,
    }


def enrich_result_metrics(
    result: dict[str, Any],
    detected_language: str,
    input_text: str,
) -> dict[str, Any]:
    """Add UI-only metric fields without changing summarization behavior."""
    result.setdefault("language", detected_language)
    result.setdefault("input_word_count", _word_count(input_text))
    result.setdefault("summary_word_count", _word_count(result.get("summary", "")))
    return result


def generate_summary_payload(
    summary_method: str,
    summary_ratio: float,
    display_text: str,
    detected_language: str,
    summarization_text: str = "",
    turkish_reduction_method: str = "textrank",
    turkish_model_key: str = "mt5",
) -> dict[str, Any]:
    """Generate a summary payload for the selected UI mode."""
    # Use summarization_text as primary input for all summarizers
    input_text = (
        summarization_text
        if summarization_text.strip() and len(summarization_text.split()) >= 100
        else display_text
    )
    if summary_method == "TF-IDF":
        result = summarize_with_tfidf(
            input_text,
            summary_ratio=summary_ratio,
            language=detected_language,
        )
        return {
            "mode": "single_extractive",
            "result": enrich_result_metrics(result, detected_language, input_text),
        }

    if summary_method == "TextRank":
        result = summarize_with_textrank(
            input_text,
            summary_ratio=summary_ratio,
            language=detected_language,
        )
        return {
            "mode": "single_extractive",
            "result": enrich_result_metrics(result, detected_language, input_text),
        }

    if summary_method == "Transformer":
        if detected_language == "en":
            transformer_result = summarize_english_transformer(input_text)
        elif detected_language == "tr":
            transformer_result = summarize_turkish_hybrid_transformer(
                input_text,
                extractive_ratio=summary_ratio,
                reduction_method=turkish_reduction_method,
                turkish_model_key=turkish_model_key,
            )
        else:
            transformer_result = summarize_with_transformer(
                input_text,
                detected_language,
            )
        return {
            "mode": "transformer",
            "result": enrich_result_metrics(
                transformer_result,
                detected_language,
                input_text,
            ),
        }

    tfidf_result = summarize_with_tfidf(
        input_text,
        summary_ratio=summary_ratio,
        language=detected_language,
    )
    textrank_result = summarize_with_textrank(
        input_text,
        summary_ratio=summary_ratio,
        language=detected_language,
    )
    tfidf_result = enrich_result_metrics(tfidf_result, detected_language, input_text)
    textrank_result = enrich_result_metrics(textrank_result, detected_language, input_text)

    if summary_method == "Compare Both":
        return {
            "mode": "compare_both",
            "tfidf_result": tfidf_result,
            "textrank_result": textrank_result,
        }

    if detected_language == "en":
        transformer_result = summarize_english_transformer(input_text)
    elif detected_language == "tr":
        transformer_result = summarize_turkish_hybrid_transformer(
            input_text,
            extractive_ratio=summary_ratio,
            reduction_method=turkish_reduction_method,
            turkish_model_key=turkish_model_key,
        )
    else:
        st.warning(t("transformer_language_warning"))
        transformer_result = summarize_with_transformer(
            input_text,
            detected_language,
        )

    return {
        "mode": "compare_all",
        "tfidf_result": tfidf_result,
        "textrank_result": textrank_result,
        "transformer_result": enrich_result_metrics(
            transformer_result,
            detected_language,
            input_text,
        ),
    }


def render_summary_payload(payload: dict[str, Any], original_text: str) -> None:
    """Render a generated summary payload without changing summarization logic."""
    mode = payload["mode"]

    if mode == "single_extractive":
        render_summary_result(payload["result"])
    elif mode == "transformer":
        render_transformer_result(payload["result"])
    elif mode == "compare_both":
        render_compare_both_results(
            payload["tfidf_result"],
            payload["textrank_result"],
            original_text,
        )
    elif mode == "compare_all":
        render_compare_all_results(
            payload["tfidf_result"],
            payload["textrank_result"],
            payload["transformer_result"],
            original_text,
        )


def generate_and_store_summary(
    uploaded_file: Any,
    summary_method: str,
    summary_ratio: float,
    turkish_reduction_method: str = "textrank",
    turkish_model_key: str = "mt5",
) -> None:
    """Run document processing and summarization with lightweight progress UI."""
    progress_bar = st.progress(0)
    status_text = st.empty()

    document = process_uploaded_pdf(
        uploaded_file,
        progress_bar=progress_bar,
        status_text=status_text,
    )
    if document is None:
        progress_bar.empty()
        status_text.empty()
        return

    progress_bar.progress(70)
    update_loading_status(status_text, t("generating_summary"))
    summary_payload = generate_summary_payload(
        summary_method,
        summary_ratio,
        str(document["display_text"]),
        str(document["detected_language"]),
        summarization_text=str(document.get("summarization_text", "")),
        turkish_reduction_method=turkish_reduction_method,
        turkish_model_key=turkish_model_key,
    )

    progress_bar.progress(95)
    update_loading_status(status_text, t("preparing_results"))
    st.session_state["document_payload"] = document
    st.session_state["summary_payload"] = summary_payload
    st.session_state["last_summary_method"] = summary_method
    st.session_state["last_summary_ratio"] = summary_ratio
    st.session_state["last_turkish_reduction_method"] = turkish_reduction_method
    st.session_state["last_turkish_model_key"] = turkish_model_key
    progress_bar.progress(100)
    update_loading_status(status_text, t("preparing_results"))
    st.rerun()


def main() -> None:
    """Run the Streamlit application."""
    st.set_page_config(
        page_title="Multilingual Article Summarizer",
        layout="wide",
    )

    inject_custom_css()

    summary_payload = st.session_state.get("summary_payload")
    document_payload = st.session_state.get("document_payload")

    if summary_payload and document_payload:
        render_hero()
        left_col, right_col = st.columns([0.24, 0.76], gap="large")

        with left_col:
            render_analysis_control_card(
                str(document_payload["file_name"]),
                str(document_payload["detected_language"]),
                document_payload["stats"],
                str(st.session_state.get("last_summary_method", "TF-IDF")),
                float(st.session_state.get("last_summary_ratio", 0.15)),
            )
            render_preprocessing_debug(document_payload["stats"])
            with st.expander(t("change_pdf")):
                uploaded_file = st.file_uploader(
                    t("upload_pdf_article"),
                    type=["pdf"],
                    key="pdf_upload",
                )
            with st.container(border=True):
                (
                    summary_method,
                    summary_ratio,
                    turkish_reduction_method,
                    turkish_model_key,
                    generate_summary,
                ) = (
                    render_summary_controls(
                        t("generate_again"),
                        detected_language=str(document_payload["detected_language"]),
                    )
                )

            if uploaded_file is not None:
                uploaded_signature = (
                    f"{uploaded_file.name}:{getattr(uploaded_file, 'size', 0)}"
                )
                if uploaded_signature != st.session_state.get("active_file_signature"):
                    st.session_state.pop("summary_payload", None)
                    st.session_state.pop("document_payload", None)
                    st.session_state["active_file_signature"] = uploaded_signature
                    st.session_state["turkish_reduction_method"] = "textrank"
                    st.session_state["turkish_model_key"] = "mt5"
                    st.rerun()

            if generate_summary and uploaded_file is not None:
                generate_and_store_summary(
                    uploaded_file,
                    summary_method,
                    summary_ratio,
                    turkish_reduction_method,
                    turkish_model_key,
                )

        with right_col:
            with st.container(border=True):
                st.subheader(t("summary_results"))
                render_summary_payload(
                    summary_payload,
                    str(
                        document_payload.get("summarization_text")
                        or document_payload["display_text"]
                    ),
                )
        return

    st.markdown('<div class="center-shell">', unsafe_allow_html=True)
    render_center_header()
    with st.container(border=True):
        (
            uploaded_file,
            summary_method,
            summary_ratio,
            turkish_reduction_method,
            turkish_model_key,
            generate_summary,
        ) = (
            render_method_controls(
                t("generate_summary"),
                disabled=st.session_state.get("pdf_upload") is None,
            )
        )

    if uploaded_file is None:
        st.markdown("</div>", unsafe_allow_html=True)
        return

    uploaded_signature = f"{uploaded_file.name}:{getattr(uploaded_file, 'size', 0)}"
    if uploaded_signature != st.session_state.get("active_file_signature"):
        st.session_state["active_file_signature"] = uploaded_signature
        st.session_state["turkish_reduction_method"] = "textrank"
        st.session_state["turkish_model_key"] = "mt5"
        st.session_state.pop("summary_payload", None)
        st.session_state.pop("document_payload", None)

    if generate_summary:
        generate_and_store_summary(
            uploaded_file,
            summary_method,
            summary_ratio,
            turkish_reduction_method,
            turkish_model_key,
        )
        return

    document = process_uploaded_pdf(uploaded_file)
    if document is None:
        st.markdown("</div>", unsafe_allow_html=True)
        return

    render_uploaded_file_card(str(document["file_name"]))
    render_document_statistics(str(document["detected_language"]), document["stats"])
    render_preprocessing_debug(document["stats"])
    render_turkish_model_control(
        str(document["detected_language"]),
        summary_method,
    )
    render_turkish_reduction_method_control(
        str(document["detected_language"]),
        summary_method,
    )
    render_text_tabs(
        str(document["extracted_text"]),
        str(document["display_text"]),
        str(document["nlp_text"]),
        summarization_text=str(document.get("summarization_text", "")),
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # TODO: Add Turkish lemmatization/preprocessing improvements before deeper model tuning.


if __name__ == "__main__":
    main()
