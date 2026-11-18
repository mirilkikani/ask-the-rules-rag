"""Step 5a / V2: measure retrieval quality for all 3 chunk sizes (no answer generation).

For every answerable test question we check if a correct page is in the top 5.

Usage:
    python src/eval_retrieval.py               # v1: search with the question only
    python src/eval_retrieval.py --translate   # V2: search with question + Gemini translation (EN<->DE)
"""
import argparse
import csv
import json
import os
import statistics
import sys
import time
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from rag import EMBED_MODEL, merge_results, retrieve, translate_query

QUESTIONS = Path("eval/questions.jsonl")
OUT_DIR = Path("eval/results")
TRANSLATIONS = OUT_DIR / "translations.json"   # cached, so Gemini is called only once per question
SIZES = [500, 1000, 1500]
K = 5
WAIT_SECONDS = 4                               # pause between Gemini calls (free tier rate limit)


def first_correct_rank(chunks, q):
    """Rank (1-5) of the first chunk from the right document and a correct page, else None."""
    for rank, c in enumerate(chunks, start=1):
        if c["doc"] == q["doc"] and c["page"] in q["correct_pages"]:
            return rank
    return None


def pct(part, whole):
    return f"{part}/{whole} ({part / whole:.0%})" if whole else "-"


def load_translations(questions):
    """Translate every question once with Gemini and cache the result in a JSON file."""
    from dotenv import load_dotenv
    from google import genai

    cache = json.loads(TRANSLATIONS.read_text(encoding="utf-8")) if TRANSLATIONS.exists() else {}
    missing = [q for q in questions if q["id"] not in cache]
    if missing:
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key or api_key.startswith("paste_"):
            sys.exit("Missing GEMINI_API_KEY in .env")
        client = genai.Client(api_key=api_key)
        print(f"Translating {len(missing)} questions with Gemini (cached afterwards)...")
        for i, q in enumerate(missing, start=1):
            for attempt in (1, 2):
                try:
                    cache[q["id"]] = translate_query(q["question"], client)
                    break
                except Exception as e:  # noqa: BLE001
                    if attempt == 2:
                        sys.exit(f"Translation failed for {q['id']}: {str(e)[:120]}")
                    print(f"   error, retry in 30 s: {str(e)[:80]}")
                    time.sleep(30)
            print(f"[{i:2}/{len(missing)}] {q['id']}: {cache[q['id']]}")
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            TRANSLATIONS.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
            time.sleep(WAIT_SECONDS)
    return cache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--translate", action="store_true", help="V2: also search with the translated question")
    args = parser.parse_args()
    suffix = "_translate" if args.translate else ""

    questions = [json.loads(line) for line in QUESTIONS.open(encoding="utf-8")]
    questions = [q for q in questions if q["type"] == "answerable"]
    print(f"{len(questions)} answerable test questions, mode: {'question + translation' if args.translate else 'question only'}\n")

    translations = load_translations(questions) if args.translate else {}

    embedder = SentenceTransformer(EMBED_MODEL)
    client = chromadb.PersistentClient(path="chroma_db")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = []

    for size in SIZES:
        collection = client.get_collection(f"chunks_{size}")
        rows = []
        for q in questions:
            t0 = time.perf_counter()
            if args.translate:
                chunks = merge_results([
                    retrieve(q["question"], embedder, collection, k=K),
                    retrieve(translations[q["id"]], embedder, collection, k=K),
                ], k=K)
            else:
                chunks = retrieve(q["question"], embedder, collection, k=K)
            ms = (time.perf_counter() - t0) * 1000
            rank = first_correct_rank(chunks, q)
            rows.append({
                "id": q["id"],
                "lang": q["lang"],
                "cross": int(q.get("cross_language", False)),
                "doc": q["doc"],
                "correct_pages": " ".join(map(str, q["correct_pages"])),
                "retrieved": " | ".join(f"{c['doc'][:6]} p.{c['page']}" for c in chunks),
                "rank": rank or "",
                "hit": int(rank is not None),
                "ms": round(ms, 1),
            })

        with (OUT_DIR / f"retrieval_{size}{suffix}.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        n = len(rows)
        ranks = [r["rank"] for r in rows if r["rank"]]
        en = [r for r in rows if r["lang"] == "en"]
        de = [r for r in rows if r["lang"] == "de"]
        xl = [r for r in rows if r["cross"]]
        same = [r for r in rows if not r["cross"]]
        summary.append({
            "chunk_size": size,
            "mode": "translate" if args.translate else "question_only",
            "hit@1": pct(sum(1 for x in ranks if x <= 1), n),
            "hit@3": pct(sum(1 for x in ranks if x <= 3), n),
            "hit@5": pct(len(ranks), n),
            "MRR": round(sum(1 / x for x in ranks) / n, 3),
            "hit@5 EN": pct(sum(r["hit"] for r in en), len(en)),
            "hit@5 DE": pct(sum(r["hit"] for r in de), len(de)),
            "hit@5 same-lang": pct(sum(r["hit"] for r in same), len(same)),
            "hit@5 cross-lang": pct(sum(r["hit"] for r in xl), len(xl)),
            "median_ms": round(statistics.median(r["ms"] for r in rows), 1),
        })

        misses = [r["id"] for r in rows if not r["hit"]]
        print(f"chunk size {size}: misses -> {', '.join(misses) or 'none'}")

    with (OUT_DIR / f"retrieval_summary{suffix}.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)

    print("\nRESULTS")
    for s in summary:
        print(s)
    print(f"\nSaved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
