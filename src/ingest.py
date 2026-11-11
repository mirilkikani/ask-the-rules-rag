"""Step 2: read PDFs (pdfplumber) and split them into chunks with page numbers and overlap.

Usage:
    python src/ingest.py --chunk-size 1000 --overlap 150
"""
import argparse
import json
import re
from pathlib import Path

import pdfplumber

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")

# Header/footer lines of EUR-Lex PDFs that repeat on every page (noise for search)
# Header/footer lines of EUR-Lex PDFs that repeat on every page (noise for search)
HEADER_PATTERNS = [
    re.compile(r"^EN$"),                                          # AI Act header, line 1
    re.compile(r"^OJ L, \d{1,2}\.\d{1,2}\.\d{4}$"),              # AI Act header, line 2
    re.compile(r"^(\d+/\d+ )?ELI: http\S+( \d+/\d+)?$"),         # AI Act footer, both sides
    re.compile(r"^.{0,30}Official Journal of the European Union.{0,30}$"),  # GDPR header
]


def remove_headers(raw: str) -> tuple[str, int]:
    """Drop header/footer lines. Return cleaned text and number of removed lines."""
    kept, removed = [], 0
    for line in raw.splitlines():
        if any(p.match(line.strip()) for p in HEADER_PATTERNS):
            removed += 1
        else:
            kept.append(line)
    return "\n".join(kept), removed


def clean_text(text: str) -> str:
    """Fix common PDF problems: soft hyphens, broken words, extra spaces."""
    text = text.replace("\u00ad", "")                  # invisible soft hyphen
    text = re.sub(r"-\n(?=[A-ZÄÖÜ])", "-", text)       # keep hyphen: "KI-\nVerordnung"
    text = re.sub(r"-\n(?=[a-zäöüß])", "", text)       # join word: "Daten-\nschutz"
    text = re.sub(r"\s+", " ", text)                   # many spaces/newlines -> one space
    return text.strip()


def split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into chunks of about chunk_size characters, cut at spaces."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):                            # do not cut inside a word
            space = text.rfind(" ", start, end)
            if space > start:
                end = space
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        new_start = end - overlap                      # step back for the overlap
        space = text.find(" ", new_start, end)         # start at a word border
        if space != -1:
            new_start = space + 1
        start = max(new_start, start + 1)              # always move forward
    return chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--overlap", type=int, default=150)
    args = parser.parse_args()

    if args.overlap >= args.chunk_size:
        raise ValueError("Overlap must be smaller than chunk size.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUT_DIR / f"chunks_{args.chunk_size}.jsonl"

    pdfs = sorted(RAW_DIR.glob("*.pdf"))
    all_lengths = []

    with out_file.open("w", encoding="utf-8") as f:
        for pdf_path in pdfs:
            lang = "de" if pdf_path.stem.endswith("_de") else "en"
            doc_chunks, doc_removed = 0, 0
            with pdfplumber.open(pdf_path) as pdf:
                n_pages = len(pdf.pages)
                for page_num, page in enumerate(pdf.pages, start=1):
                    raw, removed = remove_headers(page.extract_text() or "")
                    doc_removed += removed
                    text = clean_text(raw)
                    for i, chunk in enumerate(split_text(text, args.chunk_size, args.overlap)):
                        record = {
                            "chunk_id": f"{pdf_path.stem}_p{page_num}_c{i}",
                            "doc": pdf_path.name,
                            "lang": lang,
                            "page": page_num,
                            "text": chunk,
                        }
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        all_lengths.append(len(chunk))
                        doc_chunks += 1
            print(f"{pdf_path.name}: {n_pages} pages -> {doc_chunks} chunks "
                  f"({doc_removed} header/footer lines removed)")

    print("-" * 50)
    print(f"Chunk size {args.chunk_size}, overlap {args.overlap}")
    print(f"TOTAL chunks: {len(all_lengths)}")
    print(f"Chunk length: min {min(all_lengths)}, "
          f"avg {sum(all_lengths) // len(all_lengths)}, max {max(all_lengths)}")
    print(f"Saved to {out_file}")


if __name__ == "__main__":
    main()