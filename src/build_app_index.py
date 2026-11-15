"""Step 6: build the EU-only index files that ship with the Streamlit app.

The EU texts (AI Act, GDPR) are CC BY 4.0, so their chunks may be published.
The DSK guides are NOT included here: the app downloads them live from the DSK website.

Usage:
    python src/build_app_index.py
"""
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from rag import EMBED_MODEL

EU_DOCS = {"eu_ai_act_2024_1689_en.pdf", "gdpr_2016_679_en.pdf"}
IN_FILE = Path("data/processed/chunks_500.jsonl")
OUT_DIR = Path("app_data")


def main():
    chunks = [json.loads(line) for line in IN_FILE.open(encoding="utf-8")]
    eu = [
        {"doc": c["doc"], "lang": c["lang"], "page": c["page"], "text": c["text"]}
        for c in chunks if c["doc"] in EU_DOCS
    ]
    print(f"{len(eu)} EU chunks (of {len(chunks)} total)")

    model = SentenceTransformer(EMBED_MODEL)
    vectors = model.encode(
        ["passage: " + c["text"] for c in eu],
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)

    OUT_DIR.mkdir(exist_ok=True)
    with (OUT_DIR / "eu_chunks_500.jsonl").open("w", encoding="utf-8") as f:
        for c in eu:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    np.save(OUT_DIR / "eu_embeddings_500.npy", vectors)

    for p in sorted(OUT_DIR.iterdir()):
        print(f"{p.name}: {p.stat().st_size / 1_000_000:.1f} MB")


if __name__ == "__main__":
    main()
