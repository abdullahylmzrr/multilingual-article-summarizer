"""Streamlit app for multilingual academic article analysis."""

from __future__ import annotations

from typing import Any

import streamlit as st

from config.settings import TEXT_PREVIEW_LIMIT
from modules.abstractive_engine import summarize_english_transformer
from modules.evaluation import compare_summaries
from modules.extractive_engine import summarize_with_textrank, summarize_with_tfidf
from modules.language_detector import detect_language
from modules.pdf_reader import PDFReadError, extract_text_from_pdf
from modules.preprocessing import preprocess_text


def render_summary_result(result: dict[str, Any]) -> None:
    """Render one summarization result in the Streamlit UI."""
    method = str(result["method"])

    st.markdown(f"### {method} Summary")
    if result["summary"]:
        st.write(result["summary"])
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


def render_transformer_result(result: dict[str, Any]) -> None:
    """Render one Transformer summarization result in the Streamlit UI."""
    st.markdown("### Transformer Summary")

    error_message = result.get("error")
    chunk_errors = result.get("chunk_errors", [])

    if error_message:
        st.error(str(error_message))

    if result["summary"]:
        st.write(result["summary"])
    elif not error_message:
        st.warning("No summary was generated.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Model", result["model_name"])
    col2.metric("Chunks", result["chunk_count"])
    col3.metric("Input Words", result["input_word_count"])
    col4.metric("Summary Words", result["summary_word_count"])

    with st.expander("Chunk summaries"):
        chunk_summaries = result["chunk_summaries"]
        if chunk_summaries:
            for index, chunk_summary in enumerate(chunk_summaries, start=1):
                st.write(f"{index}. {chunk_summary}")
        else:
            st.write("No chunk summaries available.")

    if error_message:
        with st.expander("Debug details"):
            st.write(f"Error: {error_message}")
            st.write(f"Model: {result['model_name']}")
            st.write(f"Chunk count: {result['chunk_count']}")
            st.write(f"Input word count: {result['input_word_count']}")
            st.write(f"Summary word count: {result['summary_word_count']}")
            if chunk_errors:
                st.write("Chunk errors:")
                for error in chunk_errors:
                    st.write(f"- {error}")


def main() -> None:
    """Run the Streamlit application."""
    st.set_page_config(
        page_title="Multilingual Article Summarizer",
        layout="wide",
    )

    st.title("Multilingual Academic Article Summarization and Comparison System")
    st.write("Upload a Turkish or English academic PDF to extract text and detect its language.")

    uploaded_file = st.file_uploader("Upload PDF article", type=["pdf"])

    if uploaded_file is None:
        st.info("Please upload a PDF file to begin.")
        return

    st.subheader("Uploaded File")
    st.write(uploaded_file.name)

    try:
        extracted_text = extract_text_from_pdf(uploaded_file.getvalue())
    except PDFReadError as error:
        st.error(str(error))
        return

    if not extracted_text.strip():
        st.warning("No extractable text was found in this PDF.")
        return

    detected_language = detect_language(extracted_text)
    preprocessing_result = preprocess_text(extracted_text, detected_language)
    display_text = str(preprocessing_result["display_text"])
    nlp_text = str(preprocessing_result["nlp_text"])
    stats = preprocessing_result["stats"]

    if not isinstance(stats, dict):
        st.error("Preprocessing failed to produce document statistics.")
        return

    st.subheader("Document Statistics")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Detected Language", detected_language)
    col2.metric("Raw Characters", stats["raw_character_count"])
    col3.metric("Raw Words", stats["raw_word_count"])
    col4.metric("Cleaned Characters", stats["cleaned_character_count"])
    col5.metric("Cleaned Words", stats["cleaned_word_count"])

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

    st.subheader("Summarization")
    summary_method = st.selectbox(
        "Summarization method",
        options=["TF-IDF", "TextRank", "Compare Both", "Transformer"],
    )
    summary_ratio = st.selectbox(
        "Summary ratio",
        options=[0.10, 0.20, 0.30, 0.40],
        index=1,
        format_func=lambda value: f"{int(value * 100)}%",
    )

    if st.button("Generate Summary"):
        if summary_method == "TF-IDF":
            render_summary_result(
                summarize_with_tfidf(display_text, summary_ratio=summary_ratio)
            )
        elif summary_method == "TextRank":
            render_summary_result(
                summarize_with_textrank(display_text, summary_ratio=summary_ratio)
            )
        elif summary_method == "Transformer":
            if detected_language != "en":
                st.warning(
                    "Transformer summarization is currently implemented only for English. "
                    "Turkish support will be added later."
                )
            else:
                with st.spinner("Loading Transformer model and generating summary..."):
                    transformer_result = summarize_english_transformer(display_text)
                render_transformer_result(transformer_result)
        else:
            tfidf_result = summarize_with_tfidf(display_text, summary_ratio=summary_ratio)
            textrank_result = summarize_with_textrank(
                display_text,
                summary_ratio=summary_ratio,
            )
            render_compare_both_results(
                tfidf_result,
                textrank_result,
                display_text,
            )

    # TODO: Add Turkish Transformer summarization once the Turkish model flow is ready.


if __name__ == "__main__":
    main()
