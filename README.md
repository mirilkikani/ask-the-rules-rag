# Ask the rules: a measured RAG assistant for EU AI and data protection rules

![Results](docs/results_card.png)

> **Status: work in progress.** The pipeline and the evaluation are finished. A live Streamlit demo and automated tests are coming next.

## The question

Can a small, free RAG system answer questions about the **EU AI Act**, the **GDPR** and the **German data protection authorities' (DSK) guidance** with correct page citations, and **how often is it right?**

Most "chat with PDF" projects stop at a demo. This one measures retrieval and answers on a fixed test set of 60 questions.

## Results

Measured on 50 questions with known correct pages and 10 questions the system must refuse (chunk size 500, `gemini-3.5-flash-lite`, temperature 0).

| Metric | Result |
|---|---|
| Correct page in the top 5 search results (Hit@5) | **43/50 (86%)** |
| Correct page cited in the answer, when it was in the top 5 | **37/43 (86%)** |
| Correct page cited, all answerable questions | 37/50 (74%) |
| Off-topic, near-topic and prompt-injection questions refused | **10/10 (100%)** |
| Answerable questions refused ("I don't know") | 10/50 (20%) |
| Median time per answer (search + Gemini) | 1.2 s |

### Chunk size comparison (retrieval only)

| Chunk size (characters) | Chunks | Hit@1 | Hit@3 | Hit@5 | MRR |
|---|---|---|---|---|---|
| **500** | 2,471 | 23/50 (46%) | 39/50 (78%) | **43/50 (86%)** | **0.621** |
| 1000 | 1,278 | 22/50 (44%) | 35/50 (70%) | 39/50 (78%) | 0.570 |
| 1500 | 878 | 24/50 (48%) | 35/50 (70%) | 37/50 (74%) | 0.585 |

With strict first labels (before the manual review below) Hit@5 was 78% / 72% / 70%. With 50 questions, one question is 2 percentage points, so the differences between sizes are small.

### What I learned from reading the results

- **A wrong page leads to a confident wrong answer.** For "maximum fine for prohibited AI practices", the search returned the page with fines for *EU institutions* (EUR 1 500 000) instead of the general maximum (EUR 35 000 000, AI Act p.115). Gemini answered with the wrong number.
- **Cross-language search is the main weakness.** German questions about English laws (and one English question about a German guide) did not find the right page: the top 5 results were always in the question's language.
- **The system is safe but cautious.** 4 of the 10 refusals were correct (the right page was not found). 6 were over-cautious: the right page was there, but the sources only answered part of the question.
- **I found a bug in my own evaluation.** Gemini sometimes cites as `[1, 2]`; my first parser only counted `[1]`. Fixing it raised "correct page cited" from 31/50 to 37/50. The answers themselves did not change.
- **PDF extraction loses detail.** The AI Act's compute threshold "10^25" was extracted as "1025", and Gemini refused that question.

## Data and licence

| Document | Pages | Language | Licence |
|---|---|---|---|
| Regulation (EU) 2024/1689 (AI Act), original OJ text of 12.7.2024 | 144 | EN | © European Union, EUR-Lex, CC BY 4.0 |
| Regulation (EU) 2016/679 (GDPR) | 88 | EN | © European Union, EUR-Lex, CC BY 4.0 |
| DSK Orientierungshilfe "Künstliche Intelligenz und Datenschutz" (May 2024) | 15 | DE | No licence stated |
| DSK Orientierungshilfe zu RAG-Systemen (Oct 2025) | 18 | DE | No licence stated |

**Total: 265 pages, all with extractable text.** Changes made: the PDFs were converted to text and split into chunks; the text itself was not edited. The PDFs are **not** in this repository; download links are in [SOURCES.md](SOURCES.md). All documents are public; no personal data is processed.

## How it works

1. **Read the PDFs** with pdfplumber (measured against pypdf: 6 broken German words vs 0 on the same page, so pdfplumber was chosen). 518 repeated EUR-Lex header/footer lines removed.
2. **Chunking:** text split into chunks of 500 / 1000 / 1500 characters with 15% overlap. Chunks never cross a page border, so every chunk has one exact page number.
3. **Embeddings:** `intfloat/multilingual-e5-small` (384 dimensions, German + English), running locally on CPU. Building the index takes about 90 to 100 seconds per chunk size.
4. **Vector search:** ChromaDB returns the 5 most similar chunks.
5. **Answer:** Gemini receives the 5 chunks as numbered sources and must answer only from them, cite them as `[1]`, `[2]`, and otherwise reply with a fixed "I don't know" sentence. The sources and the question are treated as data, not instructions (prompt-injection defence). Questions are limited to 500 characters.

## Evaluation method

- **Test set** ([eval/questions.jsonl](eval/questions.jsonl)): 50 answerable questions (20 AI Act, 16 GDPR, 14 DSK; 38 English, 12 German, including 6 cross-language questions) plus 10 refusal questions (4 off-topic, 2 near-topic traps, 4 prompt-injection attacks).
- **Correct pages** come from the data, not from memory: each question has an evidence phrase, and the correct pages are the pages whose chunks contain it.
- **Label review:** all 11 misses were checked by hand. 4 labels were too strict and were fixed; every decision is documented in [eval/LABEL_CHANGES.md](eval/LABEL_CHANGES.md). Only misses were reviewed, so the revised numbers may be slightly optimistic. Both runs are reported above.
- **Two separate measurements:** retrieval (`src/eval_retrieval.py`, no LLM) and answers (`src/eval_answers.py`, Gemini). Raw results are in [eval/results/](eval/results/).

## Limitations

- Small, self-written test set (60 questions); results are indicative, not statistically strong.
- Cross-language retrieval fails for most mixed-language questions.
- Over-cautious refusals (6 of 50 answerable questions).
- The AI Act is the original 2024 text; later amendments are not included.
- "Page" means the PDF page number, not the number printed on the page.
- Chunks are split at page borders, so a sentence across two pages is cut.
- Free-tier Gemini: rate limits and occasional "503 overloaded" errors (handled with one retry).
- Not legal advice.

## Project structure

```
├── src/
│   ├── check_pdfs.py        # page count and text check per PDF
│   ├── compare_extract.py   # pypdf vs pdfplumber comparison
│   ├── ingest.py            # read PDFs, clean, chunk with page numbers
│   ├── build_index.py       # embeddings -> ChromaDB
│   ├── search.py            # quick top-5 search test
│   ├── rag.py               # retrieval + Gemini answer with citations
│   ├── eval_retrieval.py    # Hit@1/3/5 and MRR for 3 chunk sizes
│   └── eval_answers.py      # answer, citation and refusal evaluation
├── eval/
│   ├── questions.jsonl      # 60 test questions with correct pages
│   ├── LABEL_CHANGES.md     # documented label review
│   └── results/             # raw CSV results
├── docs/results_card.png
├── SOURCES.md               # documents, links, licences
├── requirements.txt
└── .env.example             # GEMINI_API_KEY placeholder (real key never committed)
```

## How to run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

1. Download the 4 PDFs listed in [SOURCES.md](SOURCES.md) into `data/raw/`.
2. Copy `.env.example` to `.env` and add your own Gemini API key.
3. Build the index and ask a question:

```powershell
python src/ingest.py --chunk-size 500 --overlap 75
python src/build_index.py --chunk-size 500
python src/rag.py What is AI literacy?
```

4. Run the evaluation:

```powershell
python src/eval_retrieval.py
python src/eval_answers.py --chunk-size 500
```

## Tools

Python · pdfplumber · sentence-transformers (multilingual-e5-small) · ChromaDB · Google Gemini API · python-dotenv · Git/GitHub

## Next steps

- Streamlit web app with a live demo
- Automated tests with pytest (+ GitHub Actions)
- Query translation to fix cross-language search
- A softer prompt rule for partial answers, measured against the current one
