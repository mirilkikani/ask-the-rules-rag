"""Step 5a: measure retrieval quality for all 3 chunk sizes (no Gemini, free and fast).

For every answerable test question we check if a correct page is in the top 5.

Usage:
    python src/eval_retrieval.py
"""
import csv
import json
import statistics
import time
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from rag import EMBED_MODEL, retrieve

QUESTIONS = Path("eval/questions.jsonl")
OUT_DIR = Path("eval/results")
SIZES = [500, 1000, 1500]
K = 5


def first_correct_rank(chunks, q):
    """Rank (1-5) of the first chunk from the right document and a correct page, else None."""
    for rank, c in enumerate(chunks, start=1):
        if c["doc"] == q["doc"] and c["page"] in q["correct_pages"]:
            return rank
    return None


def pct(part, whole):
    return f"{part}/{whole} ({part / whole:.0%})" if whole else "-"


def main():
    questions = [json.loads(line) for line in QUESTIONS.open(encoding="utf-8")]
    questions = [q for q in questions if q["type"] == "answerable"]
    print(f"{len(questions)} answerable test questions\n")

    embedder = SentenceTransformer(EMBED_MODEL)
    client = chromadb.PersistentClient(path="chroma_db")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = []

    for size in SIZES:
        collection = client.get_collection(f"chunks_{size}")
        rows = []
        for q in questions:
            t0 = time.perf_counter()
            chunks = retrieve(q["question"], embedder, collection, k=K)
            ms = (time.perf_counter() - t0) * 1000
            rank = first_correct_rank(chunks, q)
            rows.append({
                "id": q["id"],
                "lang": q["lang"],
                "doc": q["doc"],
                "correct_pages": " ".join(map(str, q["correct_pages"])),
                "retrieved": " | ".join(f"{c['doc'][:6]} p.{c['page']}" for c in chunks),
                "rank": rank or "",
                "hit": int(rank is not None),
                "ms": round(ms, 1),
            })

        # save one CSV per chunk size
        with (OUT_DIR / f"retrieval_{size}.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        n = len(rows)
        ranks = [r["rank"] for r in rows if r["rank"]]
        en = [r for r in rows if r["lang"] == "en"]
        de = [r for r in rows if r["lang"] == "de"]
        summary.append({
            "chunk_size": size,
            "hit@1": pct(sum(1 for x in ranks if x <= 1), n),
            "hit@3": pct(sum(1 for x in ranks if x <= 3), n),
            "hit@5": pct(len(ranks), n),
            "MRR": round(sum(1 / x for x in ranks) / n, 3),
            "hit@5 EN": pct(sum(r["hit"] for r in en), len(en)),
            "hit@5 DE": pct(sum(r["hit"] for r in de), len(de)),
            "median_ms": round(statistics.median(r["ms"] for r in rows), 1),
        })

        misses = [r["id"] for r in rows if not r["hit"]]
        print(f"chunk size {size}: misses -> {', '.join(misses) or 'none'}")

    with (OUT_DIR / "retrieval_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)

    print("\nRESULTS")
    for s in summary:
        print(s)
    print(f"\nSaved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
