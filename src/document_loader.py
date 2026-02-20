"""Document loader — extracts raw text from PDFs and plain text files."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple
import streamlit as st


def extract_text(uploaded_file) -> Tuple[str, int]:
    """
    Extract text from an uploaded file.

    Returns
    -------
    (text, page_count)
    """
    name = uploaded_file.name.lower()

    if name.endswith(".pdf"):
        return _extract_pdf(uploaded_file)
    elif name.endswith(".txt"):
        text = uploaded_file.read().decode("utf-8", errors="replace")
        return text, 1
    elif name.endswith(".md"):
        text = uploaded_file.read().decode("utf-8", errors="replace")
        return text, 1
    else:
        raise ValueError(f"Unsupported file type: {uploaded_file.name}")


def _extract_pdf(uploaded_file) -> Tuple[str, int]:
    from PyPDF2 import PdfReader
    reader = PdfReader(uploaded_file)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(f"[Page {i+1}]\n{text}")
    return "\n\n".join(pages), len(reader.pages)
