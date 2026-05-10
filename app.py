"""Streamlit app for multilingual academic article analysis."""

from __future__ import annotations

import streamlit as st

from modules.language_detector import detect_language
from modules.pdf_reader import PDFReadError, extract_text_from_pdf
from utils.text_cleaner import normalize_whitespace


def count_words(text: str) -> int:
    """Return the number of whitespace-separated words in the text."""
    return len(text.split())


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

    cleaned_text = normalize_whitespace(extracted_text)
    detected_language = detect_language(cleaned_text)

    st.subheader("Document Statistics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Detected Language", detected_language)
    col2.metric("Character Count", len(cleaned_text))
    col3.metric("Word Count", count_words(cleaned_text))

    st.subheader("Extracted Text Preview")
    if cleaned_text:
        st.text_area("Preview", cleaned_text[:3000], height=350)
    else:
        st.warning("No extractable text was found in this PDF.")

    # TODO: Add TF-IDF summarization controls and output comparison.
    # TODO: Add TextRank summarization controls and output comparison.
    # TODO: Add Transformer-based summarization controls and output comparison.


if __name__ == "__main__":
    main()
