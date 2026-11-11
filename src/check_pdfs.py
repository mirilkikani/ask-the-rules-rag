from pathlib import Path
from pypdf import PdfReader

RAW_DIR = Path("data/raw")

def main():
    pdfs = sorted(RAW_DIR.glob("*.pdf"))
    if not pdfs:
        print("No PDFs found in data/raw. Download them first.")
        return

    total = 0
    for pdf in pdfs:
        reader = PdfReader(pdf)
        pages = len(reader.pages)
        empty = sum(
            1 for p in reader.pages if not (p.extract_text() or "").strip()
        )
        total += pages
        print(f"{pdf.name}: {pages} pages, {empty} pages without text")

    print(f"TOTAL: {total} pages in {len(pdfs)} files")

if __name__ == "__main__":
    main()