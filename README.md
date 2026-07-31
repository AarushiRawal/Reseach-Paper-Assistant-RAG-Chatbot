# Research Paper RAG Chatbot

A retrieval-augmented QA system over a corpus of arXiv papers. Ask a question, and
the system retrieves the most relevant chunks from the indexed papers and answers
**only from those chunks** — with citations, a groundedness check, per-request
token/latency tracking, and a REST API around it.

Built during a GenAI internship, then hardened for production concerns (input
validation, error handling, concurrency, path-traversal defence, usage tracking).

---

## What makes it more than a vector-search demo

**Multi-modal ingestion.** Papers are not just text. Tables are extracted via a
caption-anchored strategy: each `Table N` caption anchors a band of the PDF's native
text layer, and one cached LLM call reformats it into a markdown grid — with a
groundedness sanity check so the reformatter cannot invent numbers. Figures are
rendered to PNG crops, OCR'd, and captioned by a local vision model, then stored as
searchable text *and* served back to the UI as actual images.

**Intent-routed retrieval.** `select_context_docs()` routes each question rather than
running one generic similarity search:

| Intent | Routing |
|---|---|
| `"Table 2 in the T5 paper"` | label lookup scoped to one resolved paper |
| `"Summarize the GLUE paper"` | deterministic early-page pull (abstract + intro) |
| `"show me the architecture diagram"` | `content_type`-filtered pass, paper-scoped |
| `"compare BERT and BART"` | resolve every named paper, retrieve from each |
| default | stay on the named paper, reserve a slot for its opening pages |

**Two-level hallucination defence.** Every risky numeric and proper name in a
generated answer is checked against the retrieved context. Unsupported atoms trigger
a rewrite pass; if the rewrite still fails, the system falls back rather than serve
an ungrounded figure. A deterministic escape hatch answers directly from a labelled
chunk when the model over-refuses despite correct retrieval.

**Guards before spend.** Prompt-extraction attempts and pasted external text are
rejected *before* any embedding or LLM call, so a bad request costs nothing.

---

## Architecture

```
Streamlit UI ──HTTP──► FastAPI ──► run_rag()
                                     │
              guards ────────────────┤  prompt-extraction / pasted-text  (no LLM cost)
              history rewrite ───────┤  only when the question depends on a prior turn
              select_context_docs ───┤  intent routing over ChromaDB (no LLM cost)
              answer_with_grounding ─┘  Groq generation + verification pass
```

| Layer | Choice | Why |
|---|---|---|
| Embeddings | `all-MiniLM-L6-v2` | 256-token window, fast on CPU, free |
| Vector store | ChromaDB (cosine) | local persistence, metadata filtering |
| LLM | Groq `llama-3.1-8b-instant` | 500K tokens/day free tier — ~5× the daily budget of the 70B model |
| Vision | MoonDream2 (local) | figure captioning without an API bill |
| API | FastAPI + Pydantic | validation before handler code runs |

---

## Project layout

```
app/
├── config.py               every tunable + path in one place
├── main.py                 FastAPI routes
├── schemas.py              Pydantic request models
├── ingestion/
│   ├── download.py         arXiv fetch with separate connect/read timeouts
│   ├── extract → tables.py caption-anchored table ingestion
│   ├──         figures.py  figure crops, OCR, vision captions
│   ├── vision.py           MoonDream2 + Tesseract (the only GPU-aware module)
│   ├── cache.py            caption/table cache
│   └── chunking.py         token-aware splitting + vector store build
└── rag/
    ├── store.py            lazy singletons: embeddings, Chroma, LLMs
    ├── prompts.py          every prompt template
    ├── guards.py           pre-retrieval rejection
    ├── retrieval.py        intent routing + context selection
    ├── grounding.py        atom-level answer verification
    └── chain.py            the request pipeline

scripts/ingest.py           build the index
scripts/eval_metrics.py     retrieval accuracy + latency/token percentiles
scripts/inspect_store.py    debug what's actually in the store
streamlit_app.py            frontend
run_local.py                start the API (optional --tunnel)
```

---

## Setup

**1. Install**

```bash
git clone <your-repo-url> && cd rag-chatbot
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**2. Key**

```bash
cp .env.example .env        # then paste your key from https://console.groq.com/keys
```

**3. Index the corpus** (one-off, several minutes)

```bash
python scripts/ingest.py
```

**4. Run**

```bash
python run_local.py                 # API on http://127.0.0.1:8000
streamlit run streamlit_app.py      # UI on http://localhost:8501
```

---

## GPU notes

**A GPU is optional.** Only figure captioning (MoonDream2) uses one, and it already
falls back to CPU automatically. Everything else is either tiny (MiniLM embeddings),
a remote API call (Groq generation, table reformatting), or a CPU binary (Tesseract).

Crucially, vision captioning runs at **ingestion time only** — capped at 6 figures per
paper and cached afterwards. Serving a query never touches the GPU.

| Setup | Ingestion (11 papers) | Query latency |
|---|---|---|
| CUDA GPU | fastest | unchanged — no local model in the query path |
| CPU only | slower (vision captioning dominates) | unchanged |
| CPU, vision disabled | fast | unchanged, minus figure descriptions |

For a CUDA build of PyTorch, install it from the official index *before* the rest:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

To skip vision entirely, set `IMAGE_RAG_POLICY["use_vision_caption"] = False` in
`app/config.py`. Figure chunks then keep their PDF caption and OCR text, which are
still searchable.

**Tesseract** is a system binary, not a pip package:
`sudo apt install tesseract-ocr` (Linux) / `brew install tesseract` (macOS). If it is
absent, OCR is skipped gracefully.

---

## API

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/health` | status + chunk count |
| `GET` | `/papers` | indexed corpus |
| `POST` | `/chat` | question → grounded answer, sources, usage, latency |
| `POST` | `/ingest` | add a paper by arXiv ID (idempotent) |
| `DELETE` | `/papers/{arxiv_id}` | remove chunks, PNGs, cache entries, PDF, index entry |
| `GET` | `/figure` | serve a figure crop (path-traversal guarded) |
| `POST` | `/feedback` | log a thumbs up/down |

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is self-attention?", "chat_history": []}'
```

Every `/chat` response carries `latency_ms`, `usage` (LLM calls + prompt/completion
tokens), and `mode` (`rag` or `rejected`).

---

## Testing

```bash
pytest                        # pure-logic tests: guards, routing, paper resolution
pytest -m integration         # needs ingestion done + server running
python scripts/eval_metrics.py                 # retrieval accuracy, zero LLM cost
python scripts/eval_metrics.py --with-latency  # + real generations (spends tokens)
```

---

## Known limitations

Deliberate trade-offs, not oversights:

- **No auth.** The API is unauthenticated; it is intended for local use.
- **The injection guard is a keyword regex** and is bypassable by a determined user.
- **Grounding is atom-level.** It catches fabricated numbers and names, not
  paraphrased falsehoods — an LLM judge would catch more but double the token spend.
- **`vectorstore._collection`** is a private LangChain API, used for metadata-filtered
  deletes and counts that the public wrapper doesn't expose.
- **Table reconstruction depends on the caption regex.** A table without a
  `Table N:` caption is not captured on the table path.
- **Groq free-tier rate limits** will surface as HTTP 429 during heavy demos.
