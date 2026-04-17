from __future__ import annotations

import io

from pypdf import PdfReader


def extract_text_from_pdf(pdf_bytes: bytes) -> tuple[str, int]:
    """Extract text and page count from a PDF byte stream."""
    reader = PdfReader(io.BytesIO(pdf_bytes))
    page_count = len(reader.pages)

    page_texts: list[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        page_texts.append(text)

    content = "\n".join(page_texts).strip()
    return content, page_count
