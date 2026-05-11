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
            border-bottom: 1px solid rgba(125, 211, 252, 0.12);
        }}

        [data-testid="stTabs"] button[role="tab"] {{
            height: 2.75rem;
            padding: 0 1rem;
            border-radius: 999px 999px 0 0;
            color: #adc2d7;
            background: transparent;
        }}

        [data-testid="stTabs"] button[role="tab"][aria-selected="true"] {{
            color: #f8fcff;
            background: rgba(57, 217, 255, 0.12);
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
    st.markdown(
        """
        <section class="hero-section">
            <p class="hero-kicker">Academic NLP Workspace</p>
            <h1>Multilingual Academic Article Summarizer</h1>
            <p>
                Compare TF-IDF, TextRank and Transformer-based summaries for Turkish
                and English academic PDFs.
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_center_header() -> None:
    """Render the compact centered workspace introduction."""
    st.markdown(
        """
        <div class="control-card">
            <h1>Multilingual Academic Article Summarizer</h1>
            <p>
                Upload a Turkish or English academic PDF and compare TF-IDF,
                TextRank, and Transformer summaries.
            </p>
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
) -> tuple[str, float, bool]:
    """Render method, ratio, and action controls."""
    summary_method = st.selectbox(
        "Summarization method",
        options=["TF-IDF", "TextRank", "Transformer", "Compare Both", "Compare All"],
        key="summary_method",
    )
    summary_ratio = st.selectbox(
        "Summary ratio",
        options=[0.10, 0.20, 0.30, 0.40],
        index=1,
        format_func=lambda value: f"{int(value * 100)}%",
        key="summary_ratio",
    )
    generate_clicked = st.button(
        button_label,
        disabled=disabled,
        use_container_width=True,
    )
    return str(summary_method), float(summary_ratio), generate_clicked


def render_method_controls(
    button_label: str,
    *,
    disabled: bool = False,
) -> tuple[Any, str, float, bool]:
    """Render upload and summarization controls and return their values."""
    uploaded_file = st.file_uploader("Upload PDF article", type=["pdf"], key="pdf_upload")
    summary_method, summary_ratio, generate_clicked = render_summary_controls(
        button_label,
        disabled=disabled,
    )
    return uploaded_file, str(summary_method), float(summary_ratio), generate_clicked


def render_uploaded_file_card(file_name: str) -> None:
    """Render the uploaded file name in a compact glass panel."""
    st.markdown(
        f"""
        <div class="uploaded-file-card">
            <strong>Uploaded file:</strong> {html.escape(file_name)}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_document_statistics(detected_language: str, stats: dict[str, Any]) -> None:
    """Render document statistics as compact metric cards."""
    st.subheader("Document Statistics")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Detected Language", detected_language)
    col2.metric("Raw Words", stats["raw_word_count"])
    col3.metric("Cleaned Words", stats["cleaned_word_count"])
    col4.metric("Raw Characters", stats["raw_character_count"])
    col5.metric("Cleaned Characters", stats["cleaned_character_count"])


def render_text_tabs(extracted_text: str, display_text: str, nlp_text: str) -> None:
    """Render raw, cleaned, and NLP text previews."""
    raw_tab, cleaned_tab, nlp_tab = st.tabs(["Raw Text", "Cleaned Text", "NLP Text"])

    with raw_tab:
        st.text_area(
            "Raw text preview",
            extracted_text[:TEXT_PREVIEW_LIMIT],
            height=350,
        )

    with cleaned_tab:
        st.text_area(
            "Cleaned text preview",
            display_text[:TEXT_PREVIEW_LIMIT],
            height=350,
        )

    with nlp_tab:
        st.text_area(
            "NLP text preview",
            nlp_text[:TEXT_PREVIEW_LIMIT],
            height=350,
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
                <span>File</span>
                <strong>{html.escape(file_name)}</strong>
            </div>
            <div class="side-stat">
                <span>Detected language</span>
                <strong>{html.escape(detected_language)}</strong>
            </div>
            <div class="side-stat">
                <span>Method</span>
                <strong>{html.escape(summary_method)}</strong>
            </div>
            <div class="side-stat">
                <span>Ratio</span>
                <strong>{summary_ratio:.0%}</strong>
            </div>
            <div class="side-stat">
                <span>Raw words</span>
                <strong>{stats["raw_word_count"]}</strong>
            </div>
            <div class="side-stat">
                <span>Cleaned words</span>
                <strong>{stats["cleaned_word_count"]}</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_summary_result(result: dict[str, Any]) -> None:
    """Render one summarization result in the Streamlit UI."""
    method = str(result["method"])

    st.markdown(f"### {method} Summary")
    if result["summary"]:
        render_text_card(str(result["summary"]))
    else:
        st.warning("No summary could be generated from the current text.")

    if result.get("message"):
        st.info(str(result["message"]))

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Original Sentences", result["original_sentence_count"])
    col2.metric("Valid Sentences", result["valid_sentence_count"])
    col3.metric("Selected Sentences", result["selected_sentence_count"])
    col4.metric("Summary Ratio", f"{result['summary_ratio']:.0%}")

    with st.expander(f"{method} selected sentences"):
        for index, sentence in enumerate(result["selected_sentences"], start=1):
            st.write(f"{index}. {sentence}")


def render_comparison_metrics(comparison: dict[str, Any]) -> None:
    """Render TF-IDF and TextRank comparison metrics."""
    sentence_overlap = comparison["sentence_overlap"]
    metrics_table = [
        {
            "Metric": "Original word count",
            "Value": comparison["original_word_count"],
        },
        {
            "Metric": "TF-IDF summary word count",
            "Value": comparison["tfidf_word_count"],
        },
        {
            "Metric": "TextRank summary word count",
            "Value": comparison["textrank_word_count"],
        },
        {
            "Metric": "TF-IDF compression ratio",
            "Value": f"{comparison['tfidf_compression_ratio']:.2%}",
        },
        {
            "Metric": "TextRank compression ratio",
            "Value": f"{comparison['textrank_compression_ratio']:.2%}",
        },
        {
            "Metric": "Common selected sentences",
            "Value": sentence_overlap["common_sentence_count"],
        },
        {
            "Metric": "Jaccard similarity",
            "Value": f"{sentence_overlap['jaccard_similarity']:.2%}",
        },
        {
            "Metric": "Overlap percentage",
            "Value": f"{sentence_overlap['overlap_percentage']:.2f}%",
        },
    ]

    st.markdown("### Comparison Metrics")
    st.table(metrics_table)

    with st.expander("Common selected sentences"):
        common_sentences = sentence_overlap["common_sentences"]
        if common_sentences:
            for index, sentence in enumerate(common_sentences, start=1):
                st.write(f"{index}. {sentence}")
        else:
            st.write("No common selected sentences.")


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
    metrics_table = [
        {
            "Metric": "Original word count",
            "Value": comparison["original_word_count"],
        },
        {
            "Metric": "TF-IDF summary word count",
            "Value": comparison["tfidf_word_count"],
        },
        {
            "Metric": "TextRank summary word count",
            "Value": comparison["textrank_word_count"],
        },
        {
            "Metric": "Transformer summary word count",
            "Value": comparison["transformer_word_count"],
        },
        {
            "Metric": "TF-IDF compression ratio",
            "Value": f"{comparison['tfidf_compression_ratio']:.2%}",
        },
        {
            "Metric": "TextRank compression ratio",
            "Value": f"{comparison['textrank_compression_ratio']:.2%}",
        },
        {
            "Metric": "Transformer compression ratio",
            "Value": f"{comparison['transformer_compression_ratio']:.2%}",
        },
        {
            "Metric": "TF-IDF selected sentence count",
            "Value": comparison["tfidf_selected_sentence_count"],
        },
        {
            "Metric": "TextRank selected sentence count",
            "Value": comparison["textrank_selected_sentence_count"],
        },
        {
            "Metric": "Transformer chunk count",
            "Value": comparison["transformer_chunk_count"],
        },
        {
            "Metric": "TF-IDF/TextRank common selected sentences",
            "Value": sentence_overlap["common_sentence_count"],
        },
        {
            "Metric": "TF-IDF/TextRank Jaccard similarity",
            "Value": f"{sentence_overlap['jaccard_similarity']:.2%}",
        },
        {
            "Metric": "TF-IDF/TextRank overlap percentage",
            "Value": f"{sentence_overlap['overlap_percentage']:.2f}%",
        },
    ]

    st.markdown("### All Methods Comparison Metrics")
    st.table(metrics_table)


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
    st.markdown("### Transformer Summary")

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
        st.warning("No summary was generated.")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Model", result["model_name"])
    col2.metric("Detected Language", result["language"])
    col3.metric("Chunks", result["chunk_count"])
    col4.metric("Input Words", result["input_word_count"])
    col5.metric("Summary Words", result["summary_word_count"])

    if (
        "reduced_input_word_count" in result
        or "extractive_ratio" in result
    ):
        extra_col1, extra_col2 = st.columns(2)
        if "reduced_input_word_count" in result:
            extra_col1.metric("Reduced Input Words", result["reduced_input_word_count"])
        if "extractive_ratio" in result:
            extra_col2.metric("Extractive Ratio", f"{result['extractive_ratio']:.0%}")

    with st.expander("Chunk summaries"):
        chunk_summaries = result["chunk_summaries"]
        if chunk_summaries:
            for index, chunk_summary in enumerate(chunk_summaries, start=1):
                st.write(f"{index}. {chunk_summary}")
        else:
            st.write("No chunk summaries available.")

    if (
        error_message
        or warning_message
        or rejected_chunk_summaries
        or source_filtered_sentences
    ):
        with st.expander("Debug details"):
            if error_message:
                st.write(f"Error: {error_message}")
            if warning_message:
                st.write(f"Warning: {warning_message}")
            st.write(f"Model: {result['model_name']}")
            st.write(f"Detected language: {result['language']}")
            st.write(f"Chunk count: {result['chunk_count']}")
            st.write(f"Input word count: {result['input_word_count']}")
            if "reduced_input_word_count" in result:
                st.write(f"Reduced input word count: {result['reduced_input_word_count']}")
            if "extractive_ratio" in result:
                st.write(f"Extractive ratio: {result['extractive_ratio']:.2f}")
            st.write(f"Summary word count: {result['summary_word_count']}")
            if chunk_errors:
                st.write("Chunk errors:")
                for error in chunk_errors:
                    st.write(f"- {error}")
            if rejected_chunk_summaries:
                st.write("Rejected chunk summaries:")
                for index, summary in enumerate(rejected_chunk_summaries, start=1):
                    st.write(f"{index}. {summary}")
            if source_filtered_sentences:
                st.write("Source-filtered sentences:")
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
        update_loading_status(status_text, "Reading PDF text...")

    try:
        extracted_text = extract_text_from_pdf(uploaded_file.getvalue())
    except PDFReadError as error:
        st.error(str(error))
        return None

    if not extracted_text.strip():
        st.warning("No extractable text was found in this PDF.")
        return None

    if progress_bar is not None:
        progress_bar.progress(25)
    if status_text is not None:
        update_loading_status(status_text, "Detecting language...")
    detected_language = detect_language(extracted_text)

    if progress_bar is not None:
        progress_bar.progress(45)
    if status_text is not None:
        update_loading_status(status_text, "Cleaning academic text...")
    preprocessing_result = preprocess_text(extracted_text, detected_language)
    stats = preprocessing_result["stats"]

    if not isinstance(stats, dict):
        st.error("Preprocessing failed to produce document statistics.")
        return None

    return {
        "file_name": uploaded_file.name,
        "extracted_text": extracted_text,
        "detected_language": detected_language,
        "display_text": str(preprocessing_result["display_text"]),
        "nlp_text": str(preprocessing_result["nlp_text"]),
        "stats": stats,
    }


def generate_summary_payload(
    summary_method: str,
    summary_ratio: float,
    display_text: str,
    detected_language: str,
) -> dict[str, Any]:
    """Generate a summary payload for the selected UI mode."""
    if summary_method == "TF-IDF":
        return {
            "mode": "single_extractive",
            "result": summarize_with_tfidf(display_text, summary_ratio=summary_ratio),
        }

    if summary_method == "TextRank":
        return {
            "mode": "single_extractive",
            "result": summarize_with_textrank(display_text, summary_ratio=summary_ratio),
        }

    if summary_method == "Transformer":
        if detected_language == "en":
            transformer_result = summarize_english_transformer(display_text)
        elif detected_language == "tr":
            transformer_result = summarize_turkish_hybrid_transformer(display_text)
        else:
            transformer_result = summarize_with_transformer(
                display_text,
                detected_language,
            )
        return {"mode": "transformer", "result": transformer_result}

    tfidf_result = summarize_with_tfidf(display_text, summary_ratio=summary_ratio)
    textrank_result = summarize_with_textrank(
        display_text,
        summary_ratio=summary_ratio,
    )

    if summary_method == "Compare Both":
        return {
            "mode": "compare_both",
            "tfidf_result": tfidf_result,
            "textrank_result": textrank_result,
        }

    if detected_language == "en":
        transformer_result = summarize_english_transformer(display_text)
    elif detected_language == "tr":
        transformer_result = summarize_turkish_hybrid_transformer(display_text)
    else:
        st.warning(
            "Transformer summarization requires detected language to be English or Turkish."
        )
        transformer_result = summarize_with_transformer(
            display_text,
            detected_language,
        )

    return {
        "mode": "compare_all",
        "tfidf_result": tfidf_result,
        "textrank_result": textrank_result,
        "transformer_result": transformer_result,
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
    update_loading_status(status_text, "Generating summary...")
    summary_payload = generate_summary_payload(
        summary_method,
        summary_ratio,
        str(document["display_text"]),
        str(document["detected_language"]),
    )

    progress_bar.progress(95)
    update_loading_status(status_text, "Preparing results...")
    st.session_state["document_payload"] = document
    st.session_state["summary_payload"] = summary_payload
    st.session_state["last_summary_method"] = summary_method
    st.session_state["last_summary_ratio"] = summary_ratio
    progress_bar.progress(100)
    update_loading_status(status_text, "Preparing results...")
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
                float(st.session_state.get("last_summary_ratio", 0.20)),
            )
            with st.expander("Change PDF"):
                uploaded_file = st.file_uploader(
                    "Upload PDF article",
                    type=["pdf"],
                    key="pdf_upload",
                )
            with st.container(border=True):
                summary_method, summary_ratio, generate_summary = (
                    render_summary_controls("Generate Again")
                )

            if uploaded_file is not None:
                uploaded_signature = (
                    f"{uploaded_file.name}:{getattr(uploaded_file, 'size', 0)}"
                )
                if uploaded_signature != st.session_state.get("active_file_signature"):
                    st.session_state.pop("summary_payload", None)
                    st.session_state.pop("document_payload", None)
                    st.session_state["active_file_signature"] = uploaded_signature
                    st.rerun()

            if generate_summary and uploaded_file is not None:
                generate_and_store_summary(
                    uploaded_file,
                    summary_method,
                    summary_ratio,
                )

        with right_col:
            with st.container(border=True):
                st.subheader("Summarization Results")
                render_summary_payload(
                    summary_payload,
                    str(document_payload["display_text"]),
                )
        return

    st.markdown('<div class="center-shell">', unsafe_allow_html=True)
    render_center_header()
    with st.container(border=True):
        uploaded_file, summary_method, summary_ratio, generate_summary = (
            render_method_controls(
                "Generate Summary",
                disabled=st.session_state.get("pdf_upload") is None,
            )
        )

    if uploaded_file is None:
        st.markdown("</div>", unsafe_allow_html=True)
        return

    uploaded_signature = f"{uploaded_file.name}:{getattr(uploaded_file, 'size', 0)}"
    if uploaded_signature != st.session_state.get("active_file_signature"):
        st.session_state["active_file_signature"] = uploaded_signature
        st.session_state.pop("summary_payload", None)
        st.session_state.pop("document_payload", None)

    if generate_summary:
        generate_and_store_summary(uploaded_file, summary_method, summary_ratio)
        return

    document = process_uploaded_pdf(uploaded_file)
    if document is None:
        st.markdown("</div>", unsafe_allow_html=True)
        return

    render_uploaded_file_card(str(document["file_name"]))
    render_document_statistics(str(document["detected_language"]), document["stats"])
    render_text_tabs(
        str(document["extracted_text"]),
        str(document["display_text"]),
        str(document["nlp_text"]),
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # TODO: Add Turkish lemmatization/preprocessing improvements before deeper model tuning.


if __name__ == "__main__":
    main()
