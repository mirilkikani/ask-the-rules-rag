"""Step 5b: test Gemini's answers on all 80 questions (70 answerable + 10 refusal).

Measures:
- answerable questions: did it answer (not refuse) and cite a correct page?
  (also split into same-language and cross-language questions)
- refusal questions: did it refuse correctly?
- time per question (including the translation call in V2 mode)
- answer language: is the answer in the language of the question?

Usage:
    python src/eval_answers.py                 # v1: search with the question only
    python src/eval_answers.py --translate     # V2: question + Gemini translation (EN<->DE)
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

from rag import EMBED_MODEL, GEMINI_MODEL, answer, detect_language

QUESTIONS = Path("eval/questions.jsonl")
OUT_DIR = Path("eval/results")
WAIT_SECONDS = 5          # pause between questions (free tier rate limit)


def pct(part, whole):
    return f"{part}/{whole} ({part / whole:.0%})" if whole else "-"


def ask_with_retry(q, embedder, collection, client, translate):
    """One retry after 30 s if Gemini returns an error (e.g. rate limit 429)."""
    for attempt in (1, 2):
        try:
            return answer(q, embedder, collection, client, translate=translate), ""
        except Exception as e:  # noqa: BLE001
            if attempt == 1:
                print(f"   error, retry in 30 s: {str(e)[:80]}")
                time.sleep(30)
            else:
                return None, str(e)[:200]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--translate", action="store_true", help="V2: also search with the translated question")
    args = parser.parse_args()
    mode = "translate" if args.translate else "question_only"

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key.startswith("paste_"):
        sys.exit("Missing GEMINI_API_KEY in .env")

    questions = [json.loads(line) for line in QUESTIONS.open(encoding="utf-8")]
    embedder = SentenceTransformer(EMBED_MODEL)
    collection = chromadb.PersistentClient(path="chroma_db").get_collection(f"chunks_{args.chunk_size}")
    client = genai.Client(api_key=api_key)
    print(f"{len(questions)} questions, chunk size {args.chunk_size}, mode {mode}, model {GEMINI_MODEL}\n")

    rows = []
    for i, q in enumerate(questions, start=1):
        r, error = ask_with_retry(q["question"], embedder, collection, client, args.translate)
        row = {"id": q["id"], "type": q["type"], "lang": q["lang"],
               "cross": int(bool(q.get("cross_language"))), "error": error}
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
                "seconds": round(r["translate_s"] + r["retrieval_s"] + r["generation_s"], 2),
                "translation": r["translation"].replace("\n", " "),
                "answer": r["answer"].replace("\n", " "),
                "answer_lang_ok": int(r["refused"] or detect_language(r["answer"]) == q["lang"]),
            })
        rows.append(row)
        status = "ERROR" if error else ("refused" if row["refused"] else "answered")
        print(f"[{i:2}/{len(questions)}] {q['id']}: {status}")
        time.sleep(WAIT_SECONDS)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUT_DIR / f"answers_{args.chunk_size}_{mode}.csv"
    fields = ["id", "type", "lang", "cross", "refused", "correct_page_retrieved", "correct_page_cited",
              "cited_pages", "seconds", "answer_lang_ok", "error", "translation", "answer"]
    with out_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    ok = [r for r in rows if not r["error"]]
    ans = [r for r in ok if r["type"] == "answerable"]
    ref = [r for r in ok if r["type"] == "refusal"]
    retrieved = [r for r in ans if r["correct_page_retrieved"]]
    same = [r for r in ans if not r["cross"]]
    cross = [r for r in ans if r["cross"]]

    print(f"\nRESULTS ({mode})")
    print(f"Errors: {len(rows) - len(ok)}")
    print(f"Answerable: answered (not refused)        {pct(sum(1 for r in ans if not r['refused']), len(ans))}")
    print(f"Answerable: correct page cited            {pct(sum(r['correct_page_cited'] for r in ans), len(ans))}")
    print(f"  ...when correct page was in top 5       {pct(sum(r['correct_page_cited'] for r in retrieved), len(retrieved))}")
    print(f"  ...same-language questions              {pct(sum(r['correct_page_cited'] for r in same), len(same))}")
    print(f"  ...cross-language questions             {pct(sum(r['correct_page_cited'] for r in cross), len(cross))}")
    print(f"Answerable: wrongly refused               {pct(sum(r['refused'] for r in ans), len(ans))}")
    print(f"Refusal questions: correctly refused      {pct(sum(r['refused'] for r in ref), len(ref))}")
    answered = [r for r in ans if not r["refused"]]
    print(f"Answers in the question's language        {pct(sum(r['answer_lang_ok'] for r in answered), len(answered))}")
    print("  wrong language: " + (", ".join(r["id"] for r in answered if not r["answer_lang_ok"]) or "none"))
    no_cite = [r["id"] for r in answered if "[" not in r["answer"]]
    print(f"Answers without any citation              {pct(len(no_cite), len(answered))}")
    print("  without citation: " + (", ".join(no_cite) or "none"))
    print(f"Median time per question: {statistics.median(r['seconds'] for r in ok):.2f} s")
    print(f"\nSaved to {out_file}")


if __name__ == "__main__":
    main()
