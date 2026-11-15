"""Step 5b: test Gemini's answers on all 60 questions (50 answerable + 10 refusal).

Measures:
- answerable questions: did it answer (not refuse) and cite a correct page?
- refusal questions: did it refuse correctly?
- time per question

Usage:
    python src/eval_answers.py --chunk-size 500
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
from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer

from rag import EMBED_MODEL, GEMINI_MODEL, answer

QUESTIONS = Path("eval/questions.jsonl")
OUT_DIR = Path("eval/results")
WAIT_SECONDS = 5          # pause between Gemini calls (free tier rate limit)


def pct(part, whole):
    return f"{part}/{whole} ({part / whole:.0%})" if whole else "-"


def ask_with_retry(q, embedder, collection, client):
    """One retry after 30 s if Gemini returns an error (e.g. rate limit 429)."""
    for attempt in (1, 2):
        try:
            return answer(q, embedder, collection, client), ""
        except Exception as e:  # noqa: BLE001
            if attempt == 1:
                print(f"   error, retry in 30 s: {str(e)[:80]}")
                time.sleep(30)
            else:
                return None, str(e)[:200]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-size", type=int, default=500)
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key.startswith("paste_"):
        sys.exit("Missing GEMINI_API_KEY in .env")

    questions = [json.loads(line) for line in QUESTIONS.open(encoding="utf-8")]
    embedder = SentenceTransformer(EMBED_MODEL)
    collection = chromadb.PersistentClient(path="chroma_db").get_collection(f"chunks_{args.chunk_size}")
    client = genai.Client(api_key=api_key)
    print(f"{len(questions)} questions, chunk size {args.chunk_size}, model {GEMINI_MODEL}\n")

    rows = []
    for i, q in enumerate(questions, start=1):
        r, error = ask_with_retry(q["question"], embedder, collection, client)
        row = {"id": q["id"], "type": q["type"], "lang": q["lang"], "error": error}
        if r:
            cited = [r["chunks"][n - 1] for n in r["cited"]]
            correct_cited = q["type"] == "answerable" and any(
                c["doc"] == q["doc"] and c["page"] in q["correct_pages"] for c in cited
            )
            correct_retrieved = q["type"] == "answerable" and any(
                c["doc"] == q["doc"] and c["page"] in q["correct_pages"] for c in r["chunks"]
            )
            row.update({
                "refused": int(r["refused"]),
                "correct_page_retrieved": int(correct_retrieved),
                "correct_page_cited": int(correct_cited),
                "cited_pages": " | ".join(f"{c['doc'][:12]} p.{c['page']}" for c in cited),
                "seconds": round(r["retrieval_s"] + r["generation_s"], 2),
                "answer": r["answer"].replace("\n", " "),
            })
        rows.append(row)
        status = "ERROR" if error else ("refused" if row["refused"] else "answered")
        print(f"[{i:2}/{len(questions)}] {q['id']}: {status}")
        time.sleep(WAIT_SECONDS)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUT_DIR / f"answers_{args.chunk_size}.csv"
    fields = ["id", "type", "lang", "refused", "correct_page_retrieved", "correct_page_cited",
              "cited_pages", "seconds", "error", "answer"]
    with out_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    ok = [r for r in rows if not r["error"]]
    ans = [r for r in ok if r["type"] == "answerable"]
    ref = [r for r in ok if r["type"] == "refusal"]
    retrieved = [r for r in ans if r["correct_page_retrieved"]]

    print("\nRESULTS")
    print(f"Errors: {len(rows) - len(ok)}")
    print(f"Answerable: answered (not refused)        {pct(sum(1 for r in ans if not r['refused']), len(ans))}")
    print(f"Answerable: correct page cited            {pct(sum(r['correct_page_cited'] for r in ans), len(ans))}")
    print(f"  ...when correct page was in top 5       {pct(sum(r['correct_page_cited'] for r in retrieved), len(retrieved))}")
    print(f"Answerable: wrongly refused               {pct(sum(r['refused'] for r in ans), len(ans))}")
    print(f"Refusal questions: correctly refused      {pct(sum(r['refused'] for r in ref), len(ref))}")
    print(f"Median time per question: {statistics.median(r['seconds'] for r in ok):.2f} s")
    print(f"\nSaved to {out_file}")


if __name__ == "__main__":
    main()