"""Ask the rules: Streamlit demo.

Run locally:
    streamlit run app.py
"""
import io
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pdfplumber
import streamlit as st
from google import genai
from google.genai import types
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
from ingest import clean_text, remove_headers, split_text  # noqa: E402
from rag import (  # noqa: E402
    DOC_NAMES, EMBED_MODEL, GEMINI_MODEL, MAX_QUESTION_CHARS, NOT_FOUND,
    SYSTEM_PROMPT, TOP_K, build_prompt, parse_citations,
)

APP_DATA = ROOT / "app_data"
CHUNK_SIZE, OVERLAP = 500, 75
MAX_QUESTIONS_PER_SESSION = 20          # protects the free Gemini quota
REPO_URL = "https://github.com/mirilkikani/ask-the-rules-rag"
DSK_SOURCES = {
    "dsk_oh_ki_datenschutz_2024_de.pdf":
        "https://www.datenschutzkonferenz-online.de/media/oh/20240506_DSK_Orientierungshilfe_KI_und_Datenschutz.pdf",
    "dsk_oh_rag_2025_de.pdf":
        "https://www.datenschutzkonferenz-online.de/media/oh/DSK_OH_RAG.pdf",
}
EXAMPLES = [
    "What is AI literacy?",
    "Within how many hours must a personal data breach be reported?",
    "Was ist ein RAG-System?",
]


# ---------- loading (runs once, then cached) ----------

@st.cache_resource(show_spinner="Loading the embedding model...")
def load_model():
    return SentenceTransformer(EMBED_MODEL)


def download_dsk_chunks():
    """Download the DSK guides from the official website and chunk them (same rules as ingest.py)."""
    chunks = []
    for name, url in DSK_SOURCES.items():
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (ask-the-rules demo)"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                raw, _ = remove_headers(page.extract_text() or "")
                for chunk in split_text(clean_text(raw), CHUNK_SIZE, OVERLAP):
                    chunks.append({"doc": name, "lang": "de", "page": page_num, "text": chunk})
    return chunks


@st.cache_resource(show_spinner="Building the search index (the first start can take a minute)...")
def load_index():
    model = load_model()
    chunks = [json.loads(line) for line in (APP_DATA / "eu_chunks_500.jsonl").open(encoding="utf-8")]
    vectors = np.load(APP_DATA / "eu_embeddings_500.npy")
    dsk_ok = True
    try:
        dsk = download_dsk_chunks()
        dsk_vectors = model.encode(
            ["passage: " + c["text"] for c in dsk], batch_size=32, normalize_embeddings=True
        ).astype(np.float32)
        chunks = chunks + dsk
        vectors = np.vstack([vectors, dsk_vectors])
    except Exception:  # noqa: BLE001  (no internet / site down -> EU documents only)
        dsk_ok = False
    return chunks, vectors, dsk_ok


def get_api_key():
    try:
        return st.secrets["GEMINI_API_KEY"]      # Streamlit Cloud
    except Exception:  # noqa: BLE001
        return os.getenv("GEMINI_API_KEY")       # local .env (loaded by rag.py)


@st.cache_resource
def get_client(api_key):
    """One Gemini connection, created once and kept open."""
    return genai.Client(api_key=api_key)


def retrieve(question, model, chunks, vectors, k=TOP_K):
    """Exact cosine search: vectors are normalized, so a dot product is enough."""
    q = model.encode("query: " + question, normalize_embeddings=True)
    top = np.argsort(-(vectors @ q))[:k]
    return [chunks[i] for i in top]


def set_question(text):
    st.session_state.q = text


# ---------- page ----------

st.set_page_config(page_title="Ask the rules", page_icon="⚖️", layout="centered")
st.title("Ask the rules")
st.write(
    "Ask a question about the **EU AI Act**, the **GDPR** or the German **DSK** guidance on AI "
    "and RAG systems. Answers come only from these documents, with page citations. "
    "English or German."
)
st.caption("Portfolio demo. Not legal advice.")

with st.expander("How good is it? (measured on 60 test questions)"):
    st.markdown(
        "- Correct page in the top 5 search results: **43/50 (86%)**\n"
        "- Correct page cited when it was found: **37/43 (86%)**\n"
        "- Off-topic and prompt-injection questions refused: **10/10**\n"
        "- Known weakness: questions in one language about a document in the other language\n\n"
        f"Details, test set and code: [GitHub]({REPO_URL})"
    )

model = load_model()
chunks, vectors, dsk_ok = load_index()
if not dsk_ok:
    st.warning("The DSK guides could not be downloaded right now. Answers use the EU AI Act and GDPR only.")

st.session_state.setdefault("count", 0)
st.write("Try an example:")
for col, example in zip(st.columns(len(EXAMPLES)), EXAMPLES):
    col.button(example, on_click=set_question, args=(example,), use_container_width=True)

question = st.text_input("Your question", key="q", max_chars=MAX_QUESTION_CHARS)

if st.button("Ask", type="primary") and question.strip():
    if st.session_state.count >= MAX_QUESTIONS_PER_SESSION:
        st.error(f"Limit of {MAX_QUESTIONS_PER_SESSION} questions per visit reached. Please come back later.")
        st.stop()
    api_key = get_api_key()
    if not api_key:
        st.error("No Gemini API key is configured.")
        st.stop()
    st.session_state.count += 1

    t0 = time.time()
    sources = retrieve(question.strip(), model, chunks, vectors)
    t1 = time.time()
    try:
        with st.spinner("Asking Gemini..."):
            response = get_client(api_key).models.generate_content(
                model=GEMINI_MODEL,
                contents=build_prompt(question.strip(), sources),
                config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT, temperature=0),
            )
    except Exception as e:  # noqa: BLE001
        st.error("Gemini is busy or the free limit was reached. Please try again in a minute.")
        st.caption(f"Details: {str(e)[:200]}")
        st.stop()
    t2 = time.time()

    text = (response.text or "").strip()
    if text == NOT_FOUND:
        st.info(text)
    else:
        st.markdown(text)

    cited = parse_citations(text, len(sources))
    if cited:
        st.subheader("Sources")
        for n in cited:
            c = sources[n - 1]
            with st.expander(f"[{n}] {DOC_NAMES.get(c['doc'], c['doc'])}, page {c['page']}"):
                snippet = c["text"]
                st.write(snippet[:400] + ("..." if len(snippet) > 400 else ""))
    st.caption(f"Search {t1 - t0:.2f} s · Gemini {t2 - t1:.2f} s")

st.divider()
st.caption(
    "Sources: Regulation (EU) 2024/1689 (AI Act) and Regulation (EU) 2016/679 (GDPR), "
    "© European Union, EUR-Lex, CC BY 4.0, split into chunks for search. "
    "DSK Orientierungshilfen are downloaded live from datenschutzkonferenz-online.de. "
    f"Code and evaluation: {REPO_URL}"
)
