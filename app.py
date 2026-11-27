"""Ask the rules: Streamlit demo.

Run locally:
    streamlit run app.py
"""
import html
import io
import json
import os
import re
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
# Streamlit keeps imported modules in memory between reruns. After a git push, app.py is new but an
# old rag.py could still be in memory (this caused "cannot import name 'merge_results'" on 19.11).
# Reloading makes sure the app always uses the newest files.
import importlib  # noqa: E402

import ingest  # noqa: E402
import rag  # noqa: E402

importlib.reload(ingest)
importlib.reload(rag)
from ingest import clean_text, remove_headers, split_text  # noqa: E402
from rag import (  # noqa: E402
    DOC_NAMES, EMBED_MODEL, GEMINI_MODEL, MAX_QUESTION_CHARS, NOT_FOUND,
    SYSTEM_PROMPT, TOP_K, best_snippet, build_prompt, is_refusal, merge_results, parse_citations,
    translate_query,
)

APP_DATA = ROOT / "app_data"
CHUNK_SIZE, OVERLAP = 500, 75
MAX_QUESTIONS_PER_SESSION = 20          # protects the free Gemini quota
REPO_URL = "https://github.com/mirilkikani/ask-the-rules-rag"
LINKEDIN_URL = "https://www.linkedin.com/in/mirilkikani"
DSK_SOURCES = {
    "dsk_oh_ki_datenschutz_2024_de.pdf":
        "https://www.datenschutzkonferenz-online.de/media/oh/20240506_DSK_Orientierungshilfe_KI_und_Datenschutz.pdf",
    "dsk_oh_rag_2025_de.pdf":
        "https://www.datenschutzkonferenz-online.de/media/oh/DSK_OH_RAG.pdf",
}
# Official PDFs, so every source can be opened at the exact page (#page=N)
PDF_URLS = {
    "eu_ai_act_2024_1689_en.pdf": "https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=OJ:L_202401689",
    "gdpr_2016_679_en.pdf": "https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32016R0679",
    **DSK_SOURCES,
}
# One colour per document (text colour; background is a light tint of it)
DOC_COLORS = {
    "eu_ai_act_2024_1689_en.pdf": "#1f6fd1",
    "gdpr_2016_679_en.pdf": "#14866d",
    "dsk_oh_ki_datenschutz_2024_de.pdf": "#7a4cc2",
    "dsk_oh_rag_2025_de.pdf": "#7a4cc2",
}
# (button label, full question)
EXAMPLES = [
    ("DE → AI Act: KI-Kompetenz", "Was ist KI-Kompetenz?"),   # German question, English law
    ("GDPR: breach deadline", "Within how many hours must a personal data breach be reported?"),
    ("DSK: Was ist RAG?", "Was ist ein RAG-System?"),
    ("Off-topic test", "What is the capital of France?"),
]
SUGGESTIONS = EXAMPLES[:3]

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
html, body, [class*="css"], .stMarkdown, button, input, textarea { font-family: 'Inter', sans-serif; }
.block-container, [data-testid="stMainBlockContainer"] { padding-top: 4.5rem; max-width: 860px; }
.stDeployButton, [data-testid="stAppDeployButton"] { display: none; }
h1 { letter-spacing: -0.02em; }

.eyebrow { color: #5b6573; font-size: 0.78rem; letter-spacing: 0.06em; text-transform: uppercase; margin-bottom: 0.2rem; }
.hero-title { font-size: 2.6rem; font-weight: 600; letter-spacing: -0.03em; margin: 0 0 0.4rem 0; color: #1f2328; }
.hero-sub { color: #444c56; font-size: 1.05rem; line-height: 1.6; margin-bottom: 0.8rem; }
.pill { display: inline-block; padding: 3px 11px; border-radius: 999px; font-size: 0.78rem; margin: 0 6px 6px 0; border: 1px solid; }

.stButton > button, button[data-testid="stBaseButton-secondary"] { border-radius: 999px; border: 1px solid #d0d7de; background: #ffffff; color: #1f2328; transition: all .15s ease; }
.stButton > button:hover, button[data-testid="stBaseButton-secondary"]:hover { border-color: #2a78d6; color: #2a78d6; background: #f3f8fe; }
[data-testid="stFormSubmitButton"] > button { border-radius: 10px; padding: 0.35rem 1.4rem; }
[data-testid="stTextInput"] input { border-radius: 10px; }

.via-badge { display: inline-block; margin-top: 0.55rem; padding: 2px 10px; border-radius: 999px; font-size: 0.78rem; color: #8250df; background: #8250df14; border: 1px solid #8250df40; }
.answer-card { border: 1px solid #d8dee4; border-left: 4px solid #2a78d6; border-radius: 0 12px 12px 0; padding: 1rem 1.25rem; background: #ffffff; margin: 0.6rem 0 0.4rem; font-size: 1.02rem; line-height: 1.7; color: #1f2328; }
.cite { display: inline-block; background: #e8f1fc; color: #1f6fd1; border-radius: 6px; padding: 0 6px; font-size: 0.78rem; font-weight: 500; margin-left: 2px; vertical-align: 1px; }
.meta { color: #6e7781; font-size: 0.8rem; margin-top: 0.5rem; }

.refusal-card { border: 1px solid #e3d5a8; border-left: 4px solid #c99a18; border-radius: 0 12px 12px 0; background: #fffbf0; padding: 1rem 1.25rem; margin: 0.6rem 0 0.4rem; color: #3d3423; line-height: 1.6; }
.refusal-title { font-weight: 600; margin-bottom: 0.25rem; }

.src-card { border: 1px solid #d8dee4; border-radius: 12px; padding: 0.8rem 1rem; background: #ffffff; margin-bottom: 0.75rem; }
.src-head { font-size: 0.85rem; font-weight: 600; margin-bottom: 0.35rem; }
.src-quote { color: #57606a; font-size: 0.82rem; line-height: 1.5; margin-bottom: 0.45rem; }
.src-link a { font-size: 0.82rem; color: #2a78d6; text-decoration: none; }
.src-link a:hover { text-decoration: underline; }

.stat-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin: 0.4rem 0 0.6rem; }
.stat { background: #ffffff; border: 1px solid #d8dee4; border-radius: 10px; padding: 0.55rem 0.7rem; }
.stat-num { font-size: 1.35rem; font-weight: 600; color: #1f2328; line-height: 1.2; }
.stat-label { font-size: 0.72rem; color: #5b6573; line-height: 1.3; }
.footer { color: #6e7781; font-size: 0.78rem; line-height: 1.6; }
/* ---------- animations (CSS only; switched off for "reduce motion") ---------- */
@keyframes fadeUp { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: none; } }
@keyframes pop { 0% { opacity: 0; transform: scale(0.6); } 70% { opacity: 1; transform: scale(1.08); } 100% { transform: scale(1); } }
@keyframes growLine { from { transform: scaleY(0); } to { transform: scaleY(1); } }
.hero-title, .hero-sub { animation: fadeUp .5s ease-out both; }
.hero-sub { animation-delay: .08s; }
.pill { animation: fadeUp .4s ease-out both; }
.pill:nth-child(1) { animation-delay: .15s; } .pill:nth-child(2) { animation-delay: .22s; }
.pill:nth-child(3) { animation-delay: .29s; } .pill:nth-child(4) { animation-delay: .36s; }
.stat { animation: fadeUp .45s ease-out both; }
.stat:nth-child(2) { animation-delay: .08s; } .stat:nth-child(3) { animation-delay: .16s; } .stat:nth-child(4) { animation-delay: .24s; }
.answer-card, .refusal-card { position: relative; animation: fadeUp .45s ease-out both; }
.answer-card .cite { animation: pop .35s ease-out .35s both; }
.via-badge { animation: pop .45s ease-out .55s both; }
.src-card { animation: fadeUp .45s ease-out both; transition: transform .15s ease, box-shadow .15s ease, border-color .15s ease; }
.src-card:hover { transform: translateY(-2px); box-shadow: 0 6px 18px rgba(31, 35, 40, .08); border-color: #c4ccd5; }
.stButton > button:hover, button[data-testid="stBaseButton-secondary"]:hover { transform: translateY(-1px); box-shadow: 0 4px 12px rgba(42, 120, 214, .15); }

/* page open */
@keyframes slideRight { from { opacity: 0; transform: translateX(-16px); } to { opacity: 1; transform: none; } }
@keyframes shine { from { background-position: 120% 0; } to { background-position: -20% 0; } }
@keyframes growX { from { transform: scaleX(0); } to { transform: scaleX(1); } }
[data-testid="stSidebarContent"] { animation: slideRight .55s ease-out both; }
.eyebrow { animation: fadeUp .45s ease-out both; }
.hero-title { background: linear-gradient(100deg, #1f2328 40%, #2a78d6 50%, #1f2328 60%); background-size: 250% 100%;
  -webkit-background-clip: text; background-clip: text; color: transparent;
  animation: fadeUp .5s ease-out both, shine 1.6s ease-out .4s both; }
.title-line { height: 3px; width: 64px; border-radius: 3px; background: #2a78d6; margin: -0.1rem 0 0.7rem; transform-origin: left; animation: growX .6s ease-out .3s both; }
.tagline { color: #1f2328; font-size: 1.12rem; font-weight: 500; line-height: 1.5; margin-bottom: 0.3rem; animation: fadeUp .5s ease-out .12s both; }

/* sidebar link buttons */
.side-links { display: flex; flex-wrap: wrap; gap: 8px; margin: 0.2rem 0 0.4rem; }
.side-link { display: inline-flex; align-items: center; gap: 4px; padding: 3px 11px; white-space: nowrap; border: 1px solid #d0d7de; border-radius: 999px;
  font-size: 0.85rem; color: #1f2328 !important; text-decoration: none !important; background: #ffffff;
  transition: transform .15s ease, box-shadow .15s ease, border-color .15s ease, color .15s ease; }
.side-link:hover { transform: translateY(-2px); border-color: #2a78d6; color: #2a78d6 !important; box-shadow: 0 4px 12px rgba(42, 120, 214, .15); }
.side-link:active { transform: scale(.95); }

/* clicks and focus */
.stButton > button:active, button[data-testid="stBaseButton-secondary"]:active,
[data-testid="stFormSubmitButton"] > button:active { transform: scale(.95); }
[data-testid="stFormSubmitButton"] > button { transition: transform .12s ease, box-shadow .15s ease; }
[data-testid="stFormSubmitButton"] > button:hover { box-shadow: 0 4px 14px rgba(42, 120, 214, .3); }
[data-testid="stTextInput"] input { transition: box-shadow .2s ease, border-color .2s ease; }
[data-testid="stTextInput"] input:focus { box-shadow: 0 0 0 4px rgba(42, 120, 214, .15); }
[data-testid="stExpander"] summary { transition: color .15s ease; }
[data-testid="stExpander"] summary:hover { color: #2a78d6; }
[data-testid="stExpanderDetails"] { animation: fadeUp .35s ease-out both; }

/* while scrolling (Chrome, Edge, Safari; other browsers simply show everything) */
@keyframes scrollIn { from { opacity: 0; transform: translateY(24px) scale(.98); } to { opacity: 1; transform: none; } }
.scroll-progress { display: none; }
@supports (animation-timeline: view()) {
  /* every block in the page and the sidebar slides in when it comes into view */
  [data-testid="stMain"] [data-testid="stElementContainer"],
  [data-testid="stSidebarContent"] [data-testid="stElementContainer"],
  [data-testid="stExpander"], .stat {
    animation: scrollIn linear both; animation-timeline: view(); animation-range: entry 0% cover 18%;
  }
  [data-testid="stElementContainer"]:has(.scroll-progress) { animation: none; }   /* keep the bar fixed */
  /* thin progress bar at the top that fills while scrolling down */
  [data-testid="stMain"] { scroll-timeline: --page block; }
  .scroll-progress { display: block; position: fixed; top: 0; left: 0; right: 0; height: 3px; z-index: 999990;
    background: linear-gradient(90deg, #2a78d6, #8250df); transform-origin: left; transform: scaleX(0);
    animation: growX linear both; animation-timeline: --page; }
}
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { animation: none !important; transition: none !important; }
}
</style>
"""


# ---------- loading (runs once, then cached) ----------

@st.cache_resource(show_spinner="Starting up: loading the language model (about 1 minute after a break)...")
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


@st.cache_resource(show_spinner="Starting up: downloading the German DSK guides and building the index...")
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
    """Exact cosine search: vectors are normalized, so a dot product is the similarity."""
    q = model.encode("query: " + question, normalize_embeddings=True)
    scores = vectors @ q
    top = np.argsort(-scores)[:k]
    return [{**chunks[i], "score": float(scores[i])} for i in top]


def search_both_languages(question, client, model, chunks, vectors):
    """V2: translate the question EN<->DE, search with both versions, keep the best 5 chunks.

    If the translation call fails, search with the original question only (the app keeps working).
    """
    try:
        translated = translate_query(question, client)
    except Exception:  # noqa: BLE001
        translated = ""
    found = [{**c, "via": "your question"} for c in retrieve(question, model, chunks, vectors)]
    if translated and translated.lower() != question.lower():
        found += [{**c, "via": "translation"} for c in retrieve(translated, model, chunks, vectors)]
    return merge_results([found]), translated


@st.cache_data(ttl=24 * 3600, max_entries=200, show_spinner=False)
def run_pipeline(question, prompt_version, _client, _model, _chunks, _vectors):
    """Translate, search and answer. Cached for 24 h: the same question (e.g. an example button)
    is answered instantly and does not use the free Gemini quota again. Errors are not cached.
    prompt_version (the system prompt text) is part of the cache key, so a prompt change after a
    push starts with fresh answers (on 25.11 an old uncited answer stayed cached until a reboot)."""
    t0 = time.time()
    sources, translated = search_both_languages(question, _client, _model, _chunks, _vectors)
    t1 = time.time()
    response = _client.models.generate_content(
        model=GEMINI_MODEL,
        contents=build_prompt(question, sources),
        config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT, temperature=0),
    )
    t2 = time.time()
    return {"sources": sources, "translated": translated, "text": (response.text or "").strip(),
            "search_s": t1 - t0, "answer_s": t2 - t1}


def pdf_link(chunk):
    url = PDF_URLS.get(chunk["doc"])
    return f"{url}#page={chunk['page']}" if url else None


def ask_example(question):
    st.session_state.q = question
    st.session_state.run_example = True


def pill(text, color):
    return f"<span class='pill' style='color:{color};background:{color}14;border-color:{color}40'>{text}</span>"


def answer_html(text):
    """Escape the model's text (safety), then turn [1] / [1, 2] into small citation chips."""
    safe = html.escape(text).replace("\n", "<br>")
    safe = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", safe)            # keep **bold** from the model
    return re.sub(r"\[([\d,\s]+)\]", lambda m: "".join(
        f"<span class='cite'>{n}</span>" for n in re.findall(r"\d+", m.group(1))), safe)


# ---------- page setup ----------

st.set_page_config(page_title="Ask the rules", page_icon="⚖️", layout="centered")
st.markdown(CSS, unsafe_allow_html=True)
st.session_state.setdefault("count", 0)

with st.sidebar:
    st.markdown("### Built by Miril Kikani")
    st.markdown(
        f"<div class='side-links'><a class='side-link' href='{LINKEDIN_URL}' target='_blank'>LinkedIn ↗</a>"
        f"<a class='side-link' href='{REPO_URL}' target='_blank'>GitHub repo ↗</a></div>",
        unsafe_allow_html=True,
    )
    st.caption("M.Sc. Big Data and Business Analytics, FOM Essen")

    st.markdown("### How it works")
    st.markdown(
        "1. **Read** 265 pages of PDFs and split them into chunks with page numbers\n"
        "2. **Embed** each chunk with a multilingual model (German + English)\n"
        "3. **Translate** your question (English ↔ German) and **search** with both versions\n"
        "4. **Answer** with Gemini, only from those chunks, with page citations"
    )

    st.markdown("### Measured quality")
    st.caption("80 test questions (70 with known correct pages, 10 the bot must refuse)")
    st.markdown(
        "<div class='stat-grid'>"
        "<div class='stat'><div class='stat-num'>91%</div><div class='stat-label'>correct page in top 5 (64/70)</div></div>"
        "<div class='stat'><div class='stat-num'>80–86%</div><div class='stat-label'>correct page cited (2 runs)</div></div>"
        "<div class='stat'><div class='stat-num'>10/10</div><div class='stat-label'>off-topic and attack questions refused</div></div>"
        "<div class='stat'><div class='stat-num'>8/8</div><div class='stat-label'>hidden attacks in documents resisted</div></div>"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("### Known limits")
    st.markdown(
        "- Two Gemini calls per question (translation + answer), so answers take a bit longer\n"
        "- Sometimes refuses although the answer is there (about 1 in 10 test questions)\n"
        "- AI Act in its original 2024 version\n"
        "- Portfolio demo, not legal advice"
    )

# ---------- header ----------

st.markdown("<div class='scroll-progress'></div>", unsafe_allow_html=True)
st.markdown(
    "<div class='eyebrow'>RAG assistant · EU AI Act · GDPR · DSK</div>"
    "<div class='hero-title'>Ask the rules</div>"
    "<div class='title-line'></div>"
    "<div class='tagline'>EU AI Act &amp; GDPR answers with page citations, in English and German</div>"
    "<div class='hero-sub'>Answers come only from the official documents and show the page they are "
    "based on.</div>"
    + pill("265 pages", "#1f6fd1") + pill("English and German", "#1f6fd1")
    + pill("91% correct page in top 5", "#14866d") + pill("10/10 attacks refused", "#14866d"),
    unsafe_allow_html=True,
)

model = load_model()
chunks, vectors, dsk_ok = load_index()
if not dsk_ok:
    st.warning("The DSK guides could not be downloaded right now. Answers use the EU AI Act and GDPR only.")

# ---------- examples + question ----------

st.caption("Try an example")
for col, (label, question_text) in zip(st.columns(len(EXAMPLES)), EXAMPLES):
    col.button(label, help=question_text, on_click=ask_example, args=(question_text,), width="stretch")

with st.form("ask_form", clear_on_submit=False, border=False):
    st.text_input("Your question", key="q", max_chars=MAX_QUESTION_CHARS,
                  placeholder="Within how many hours must a data breach be reported?")
    submitted = st.form_submit_button("Ask", type="primary")

run = submitted or st.session_state.pop("run_example", False)
question = st.session_state.get("q", "").strip()
will_ask = run and bool(question) and st.session_state.count < MAX_QUESTIONS_PER_SESSION

left = MAX_QUESTIONS_PER_SESSION - st.session_state.count - (1 if will_ask else 0)
st.caption(f"Press Enter to ask · {left} of {MAX_QUESTIONS_PER_SESSION} questions left in this visit")

# ---------- answer ----------

if run and question:
    if st.session_state.count >= MAX_QUESTIONS_PER_SESSION:
        st.error(f"You reached the limit of {MAX_QUESTIONS_PER_SESSION} questions for this visit. Please come back later.")
        st.stop()
    api_key = get_api_key()
    if not api_key:
        st.error("No Gemini API key is configured.")
        st.stop()
    st.session_state.count += 1

    client = get_client(api_key)
    try:
        with st.spinner("Translating, searching the documents and writing the answer..."):
            started = time.time()
            result = run_pipeline(question, SYSTEM_PROMPT, client, model, chunks, vectors)
            from_cache = time.time() - started < 0.05      # a real Gemini call takes much longer
    except Exception as e:  # noqa: BLE001
        st.error("Gemini is busy or the free limit was reached. Please try again in a minute.")
        st.caption(f"Details: {str(e)[:200]}")
        st.stop()

    sources, translated, text = result["sources"], result["translated"], result["text"]
    cited = parse_citations(text, len(sources))
    timing = f"Translate + search {result['search_s']:.2f} s · Answer {result['answer_s']:.2f} s"
    if from_cache:
        timing = "Answered instantly (same question asked before) · first time: " + timing
    via_translation = any(sources[n - 1]["via"] == "translation" for n in cited)
    badge = ("<div class='via-badge'>🌐 Found thanks to the English ↔ German translation</div>"
             if via_translation and not is_refusal(text) else "")

    if is_refusal(text):
        st.markdown(
            "<div class='refusal-card'><div class='refusal-title'>Not covered by these documents</div>"
            "I only answer from the EU AI Act, the GDPR and two German DSK guides, and I couldn't find "
            "this there. That's on purpose: no answer is better than a made-up one."
            f"<div class='meta'>{timing}</div></div>",
            unsafe_allow_html=True,
        )
        st.caption("Try one of these")
        for col, (label, question_text) in zip(st.columns(len(SUGGESTIONS)), SUGGESTIONS):
            col.button(label, key=f"suggest_{label}", help=question_text,
                       on_click=ask_example, args=(question_text,), width="stretch")
    else:
        st.markdown(f"<div class='answer-card'>{answer_html(text)}{badge}<div class='meta'>{timing}</div></div>",
                    unsafe_allow_html=True)

    shown = cited
    if not cited and not is_refusal(text):
        # Safety net: the model sometimes forgets the [n] numbers. Show the most similar pages instead.
        shown = [1, 2, 3][:len(sources)]
    if shown:
        st.markdown("<h4 class='scroll-in'>Sources</h4>", unsafe_allow_html=True)
        if not cited:
            st.caption("The answer has no citation numbers, so these are the 3 most similar pages it was based on.")
        cols = st.columns(2)
        for i, n in enumerate(shown):
            c = sources[n - 1]
            color = DOC_COLORS.get(c["doc"], "#57606a")
            name = html.escape(DOC_NAMES.get(c["doc"], c["doc"]))
            snippet = html.escape(best_snippet(c["text"], [question, translated]))
            link = pdf_link(c)
            link_html = (f"<div class='src-link'><a href='{link}' target='_blank'>"
                         f"Open page {c['page']} in the official PDF ↗</a></div>") if link else ""
            cols[i % 2].markdown(
                f"<div class='src-card' style='animation-delay:{0.25 + i * 0.12:.2f}s'>"   # one after another
                f"<div class='src-head'><span class='cite'>{n}</span> "
                f"<span style='color:{color}'>{name}</span> · page {c['page']}</div>"
                f"<div class='src-quote'>\"{snippet}\"</div>{link_html}</div>",
                unsafe_allow_html=True,
            )

    with st.expander("How this answer was found"):
        if translated:
            st.markdown(f"**Also searched as:** {translated}")
        else:
            st.caption("Translation was not available, so only your original question was searched.")
        st.caption("The 5 chunks with the highest similarity, from both searches together. "
                   "Gemini only sees these 5 and must cite them.")
        st.dataframe(
            [
                {
                    "Rank": r,
                    "Document": DOC_NAMES.get(c["doc"], c["doc"]),
                    "Page": c["page"],
                    "Similarity": round(c["score"], 3),
                    "Found by": c["via"],
                    "Cited": "yes" if r in cited else "",
                }
                for r, c in enumerate(sources, start=1)
            ],
            hide_index=True,
            width="stretch",
        )

# ---------- footer ----------

st.divider()
st.markdown(
    "<div class='footer'>Sources: Regulation (EU) 2024/1689 (AI Act) and Regulation (EU) 2016/679 (GDPR), "
    "© European Union, EUR-Lex, CC BY 4.0, split into chunks for search. DSK Orientierungshilfen are "
    "downloaded live from datenschutzkonferenz-online.de. "
    f"Code and evaluation: <a href='{REPO_URL}' target='_blank'>GitHub</a></div>",
    unsafe_allow_html=True,
)
