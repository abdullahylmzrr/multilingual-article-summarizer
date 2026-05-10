# Multilingual Academic Article Summarization and Comparison System

This Python NLP course project is an early-stage system for processing Turkish and English academic PDF articles.

The current skeleton supports:

- Uploading a PDF article through a Streamlit interface
- Extracting text from the PDF with PyMuPDF
- Detecting whether the extracted text is Turkish or English with `langdetect`
- Displaying a text preview, detected language, character count, and word count

Future work will compare:

- TF-IDF based extractive summarization
- TextRank based extractive summarization
- Transformer based abstractive summarization

## Project Structure

```text
multilingual-article-summarizer/
|-- app.py
|-- README.md
|-- requirements.txt
|-- .gitignore
|-- config/
|-- modules/
|-- pipelines/
|-- utils/
|-- data/
`-- models/
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

## Status

Summarization is intentionally not implemented yet. The project currently focuses on the foundation: PDF ingestion, text extraction, and language detection.
