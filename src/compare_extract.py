"""Compare pypdf and pdfplumber text extraction on the same pages."""
import re
from pathlib import Path

import pdfplumber
from pypdf import PdfReader

TESTS = [
    ("data/raw/dsk_oh_ki_datenschutz_2024_de.pdf", 1),
    ("data/raw/dsk_oh_rag_2025_de.pdf", 2),
    ("data/raw/eu_ai_act_2024_1689_en.pdf", 50),
]


def broken_hint_count(text: str) -> int:
    """Count ' -Word' patterns, a typical sign of wrong spaces."""
    return len(re.findall(r"\w -\w", text))


for path, page_num in TESTS:
    name = Path(path).name
    t1 = PdfReader(path).pages[page_num - 1].extract_text() or ""
    with pdfplumber.open(path) as pdf:
        t2 = pdf.pages[page_num - 1].extract_text() or ""

    print("=" * 70)
    print(f"{name}, page {page_num}")
    print(f"pypdf:      {broken_hint_count(t1)} ' -' errors")
    print(f"pdfplumber: {broken_hint_count(t2)} ' -' errors")
    print("--- pypdf (first 400 chars) ---")
    print(" ".join(t1.split())[:400])
    print("--- pdfplumber (first 400 chars) ---")
    print(" ".join(t2.split())[:400])