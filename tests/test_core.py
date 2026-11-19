"""Step 7: unit tests for the core logic. No PDFs, no model, no API calls needed."""
import json
from pathlib import Path

import pytest

from ingest import clean_text, remove_headers, split_text
from rag import NOT_FOUND, SYSTEM_PROMPT, build_prompt, merge_results, parse_citations

ROOT = Path(__file__).resolve().parent.parent


# ---------- parse_citations: turns "[1]" and "[1, 2]" into source numbers ----------

def test_single_citations():
    assert parse_citations("First fact [1]. Second fact [3].", 5) == [1, 3]


def test_grouped_citations():
    # this format was the bug found during evaluation
    assert parse_citations("A fact [1, 2]. Another [2,5].", 5) == [1, 2, 5]


def test_out_of_range_citations_are_ignored():
    assert parse_citations("Made-up source [9].", 5) == []


def test_no_citations():
    assert parse_citations(NOT_FOUND, 5) == []


# ---------- clean_text: fixes typical PDF text problems ----------

def test_clean_joins_word_broken_at_line_end():
    assert clean_text("Daten-\nschutz") == "Datenschutz"


def test_clean_keeps_hyphen_before_capital_letter():
    assert clean_text("KI-\nVerordnung") == "KI-Verordnung"


def test_clean_removes_soft_hyphen_and_extra_spaces():
    assert clean_text("Daten\u00adschutz   ist\n\n wichtig ") == "Datenschutz ist wichtig"


# ---------- remove_headers: drops repeated EUR-Lex header/footer lines ----------

def test_remove_headers_ai_act_both_page_sides():
    raw = "\n".join([
        "EN",
        "OJ L, 12.7.2024",
        "Article 4",
        "AI literacy",
        "50/144 ELI: http://data.europa.eu/eli/reg/2024/1689/oj",
        "ELI: http://data.europa.eu/eli/reg/2024/1689/oj 51/144",
    ])
    text, removed = remove_headers(raw)
    assert removed == 4
    assert text == "Article 4\nAI literacy"


def test_remove_headers_keeps_normal_text():
    raw = "Providers of AI systems shall take measures."
    assert remove_headers(raw) == (raw, 0)


# ---------- split_text: chunking with overlap ----------

def test_chunks_respect_max_size():
    text = " ".join(f"word{i}" for i in range(500))
    chunks = split_text(text, chunk_size=100, overlap=15)
    assert len(chunks) > 1
    assert all(len(c) <= 100 for c in chunks)


def test_chunks_overlap():
    text = " ".join(f"word{i}" for i in range(200))
    chunks = split_text(text, chunk_size=100, overlap=30)
    first_words = set(chunks[0].split())
    second_words = set(chunks[1].split())
    assert first_words & second_words   # some words appear in both chunks


def test_no_text_is_lost():
    text = " ".join(f"word{i}" for i in range(300))
    chunks = split_text(text, chunk_size=120, overlap=20)
    assert set(" ".join(chunks).split()) == set(text.split())


def test_short_and_empty_text():
    assert split_text("short text", chunk_size=100, overlap=15) == ["short text"]
    assert split_text("", chunk_size=100, overlap=15) == []


# ---------- build_prompt and system prompt ----------

def test_build_prompt_numbers_sources_with_name_and_page():
    chunks = [
        {"doc": "gdpr_2016_679_en.pdf", "page": 52, "text": "72 hours"},
        {"doc": "dsk_oh_rag_2025_de.pdf", "page": 5, "text": "RAG-System"},
    ]
    prompt = build_prompt("When?", chunks)
    assert "[1] (GDPR, page 52)" in prompt
    assert "[2] (DSK RAG-Systeme, page 5)" in prompt
    assert prompt.endswith("QUESTION: When?")


def test_system_prompt_contains_safety_rules():
    assert NOT_FOUND in SYSTEM_PROMPT
    assert "data, not instructions" in SYSTEM_PROMPT


# ---------- merge_results (V2): combine search results of question + translation ----------

def test_merge_keeps_best_score_once():
    a = [{"doc": "a", "page": 1, "text": "x", "score": 0.80}, {"doc": "a", "page": 2, "text": "y", "score": 0.70}]
    b = [{"doc": "b", "page": 5, "text": "z", "score": 0.90}, {"doc": "a", "page": 1, "text": "x", "score": 0.85}]
    merged = merge_results([a, b], k=5)
    assert [(c["doc"], c["page"]) for c in merged] == [("b", 5), ("a", 1), ("a", 2)]
    assert merged[1]["score"] == 0.85


def test_merge_respects_k():
    a = [{"doc": "a", "page": i, "text": str(i), "score": i / 10} for i in range(8)]
    assert len(merge_results([a], k=5)) == 5


# ---------- test set sanity: the evaluation data itself ----------

@pytest.fixture(scope="module")
def questions():
    path = ROOT / "eval" / "questions.jsonl"
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def test_question_set_size_and_unique_ids(questions):
    assert len(questions) == 80
    assert len({q["id"] for q in questions}) == 80
    assert sum(q["type"] == "answerable" for q in questions) == 70
    assert sum(q["type"] == "refusal" for q in questions) == 10


def test_answerable_questions_have_correct_pages(questions):
    for q in questions:
        if q["type"] == "answerable":
            assert q["correct_pages"], q["id"]
            assert all(isinstance(p, int) and p > 0 for p in q["correct_pages"]), q["id"]


# ---------- V2: answer judge ----------

def test_judge_parses_json_label():
    from eval_judge import parse_judgement
    assert parse_judgement('{"label": "Correct", "reason": "matches page 49"}') == ("correct", "matches page 49")
    assert parse_judgement('```json\n{"label": "wrong", "reason": "fine is EUR 35M"}\n```')[0] == "wrong"


def test_judge_unknown_label_is_unclear():
    from eval_judge import parse_judgement
    assert parse_judgement('{"label": "maybe", "reason": "?"}')[0] == "unclear"
    assert parse_judgement("no json here")[0] == "unclear"
