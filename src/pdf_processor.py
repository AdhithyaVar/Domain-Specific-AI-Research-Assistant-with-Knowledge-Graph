# src/pdf_processor.py
"""
PDF Processor — Extracts and structures text from academic PDFs using PyMuPDF.
Identifies sections like Abstract, Introduction, Methodology, Results, Conclusion.
"""

import fitz  # PyMuPDF
import re
import os
from typing import List, Dict


SECTION_PATTERNS = [
    r"abstract", r"introduction", r"background", r"related work",
    r"methodology", r"method", r"approach", r"model",
    r"experiment", r"evaluation", r"results", r"discussion",
    r"conclusion", r"future work", r"references", r"acknowledgement"
]

SECTION_REGEX = re.compile(
    r"^(" + "|".join(SECTION_PATTERNS) + r")\b",
    re.IGNORECASE
)


def extract_text_by_section(pdf_path: str) -> Dict[str, str]:
    """
    Extracts text from a PDF and groups it by detected section headings.
    Returns a dict: {section_name: section_text}
    """
    doc = fitz.open(pdf_path)
    sections: Dict[str, str] = {}
    current_section = "preamble"
    buffer = []

    for page in doc:
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                line_text = " ".join(
                    span["text"] for span in line.get("spans", [])
                ).strip()

                if not line_text:
                    continue

                # Detect section headings by font size + pattern
                is_heading = False
                for span in line.get("spans", []):
                    if span.get("size", 0) >= 11 and SECTION_REGEX.match(line_text):
                        is_heading = True
                        break

                if is_heading:
                    # Save buffer under current section
                    if buffer:
                        sections[current_section] = sections.get(
                            current_section, ""
                        ) + " ".join(buffer) + " "
                        buffer = []
                    current_section = line_text.lower().strip()
                else:
                    buffer.append(line_text)

    # Flush last buffer
    if buffer:
        sections[current_section] = sections.get(
            current_section, ""
        ) + " ".join(buffer)

    doc.close()
    return sections


def extract_full_text(pdf_path: str) -> str:
    """Returns complete raw text from the PDF."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text


def get_pdf_metadata(pdf_path: str) -> Dict:
    """Extracts basic metadata from PDF."""
    doc = fitz.open(pdf_path)
    meta = doc.metadata
    doc.close()
    return {
        "title":    meta.get("title", os.path.basename(pdf_path)),
        "author":   meta.get("author", "Unknown"),
        "subject":  meta.get("subject", ""),
        "pages":    doc.page_count if not doc.is_closed else 0,
        "filename": os.path.basename(pdf_path),
        "path":     pdf_path,
    }
