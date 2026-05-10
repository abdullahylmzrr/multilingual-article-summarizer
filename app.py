"""Streamlit app for multilingual academic article analysis."""

from __future__ import annotations

import streamlit as st

from config.settings import TEXT_PREVIEW_LIMIT
from modules.language_detector import detect_language
from modules.pdf_reader import PDFReadError, extract_text_from_pdf
from modules.preprocessing import preprocess_text


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

    # TODO: Add TF-IDF summarization controls and output comparison.
    # TODO: Add TextRank summarization controls and output comparison.
    # TODO: Add Transformer-based summarization controls and output comparison.


if __name__ == "__main__":
    main()
