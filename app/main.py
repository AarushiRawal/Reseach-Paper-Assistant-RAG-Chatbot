"""FastAPI application.

Error policy: unexpected exceptions are logged HERE with a full traceback for us
to debug, while the client only ever sees a short generic message -- never an
internal path, stack frame, or library name.
"""

import glob
import json
import logging
import os
import re
import threading
import time
import traceback

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from langchain_core.messages import AIMessage, HumanMessage

from app.config import (
    CORS_ORIGINS,
    FEEDBACK_LOG_PATH,
    PAPERS_INDEX_PATH,
    PDF_DIR,
)
from app.ingestion.cache import load_cache, save_cache
from app.ingestion.chunking import build_vectorstore, load_and_chunk
from app.ingestion.download import fetch_papers
from app.rag.chain import run_rag
from app.rag.store import count_chunks, get_vectorstore
from app.schemas import ChatRequest, FeedbackRequest, IngestRequest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag_api")

# One lock around every endpoint that MUTATES shared state (Chroma + the json
# index + files on disk). Without it, two overlapping /ingest or /delete calls
# could interleave their read-modify-write steps and corrupt papers_index.json.
# A single threading.Lock is enough at this scale -- ingestion is rare.
_mutate_lock = threading.Lock()

# Strict arXiv id shape (e.g. 1706.03762). Validating this BEFORE any file or
# DB operation means garbage ids can never reach os.remove()/Chroma filters.
ARXIV_ID_RE = re.compile(r"^\d{4}\.\d{4,5}$")


def _http_error(e, where):
    """Map an unexpected exception to a safe HTTP response.
    Groq free-tier rate limits (429) get their own friendly message because
    they WILL happen during demos; everything else is a generic 500."""
    msg = str(e).lower()
    logger.error("Error in %s:\n%s", where, traceback.format_exc())
    if "rate limit" in msg or "429" in msg:
        return HTTPException(status_code=429,
                             detail="The language model is rate-limited right now "
                                    "(free tier). Please retry in a minute.")
    return HTTPException(status_code=500,
                         detail="Internal server error. Please try again.")


app = FastAPI(title="Research Paper RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- helpers -----------------------------------------------------------------

def to_langchain_history(history):
    """Convert Streamlit-style dicts to LangChain message objects."""
    messages = []
    for msg in history:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
    return messages


def extract_sources(docs):
    """Deduplicate and format source citations from retrieved chunks."""
    seen = set()
    sources = []
    for doc in docs:
        arxiv_id = doc.metadata.get("arxiv_id")
        page = doc.metadata.get("page", "?")
        content_type = doc.metadata.get("content_type", "text")
        key = (arxiv_id, page, content_type, doc.metadata.get("source", ""))
        if key not in seen:
            seen.add(key)
            sources.append({
                "title": doc.metadata.get("title", ""),
                "arxiv_id": arxiv_id,
                "page": page,
                "content_type": content_type,
                "source": doc.metadata.get("source", ""),
                "image_path": doc.metadata.get("image_path"),  # None for text/table chunks
            })
    return sources


def _load_index():
    try:
        with open(PAPERS_INDEX_PATH) as f:
            return json.load(f)
    except FileNotFoundError:
        return []


def _save_index(index):
    with open(PAPERS_INDEX_PATH, "w") as f:
        json.dump(index, f, indent=2)


# --- endpoints ---------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "chunks": count_chunks()}


@app.get("/papers")
def list_papers():
    index = _load_index()
    if not index:
        raise HTTPException(status_code=404,
                            detail="No papers indexed yet - run scripts/ingest.py first")
    return index


@app.post("/chat")
def chat(req: ChatRequest):
    try:
        question = req.question.strip()
        if not question:
            raise HTTPException(status_code=422, detail="Question cannot be empty.")

        history = to_langchain_history(req.chat_history)
        result = run_rag(question, history, debug=False, regenerate=req.regenerate)

        return {
            "answer": result["answer"],
            "sources": extract_sources(result["context"]),
            "usage": result.get("usage", {}),
            "latency_ms": result.get("latency_ms"),
            "mode": result.get("mode", "rag"),   # "rag" or "rejected"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise _http_error(e, "/chat")


@app.get("/figure")
def figure(path: str):
    """Serve a saved figure crop so the frontend can DISPLAY the figure, not
    just its description.

    Security: the path arrives from the client, so it is resolved and checked to
    sit inside PDF_DIR before opening. Without that check a crafted path such as
    ../../etc/passwd would read arbitrary files off the server.
    """
    try:
        base = os.path.realpath(PDF_DIR)
        target = os.path.realpath(path)
        if not target.startswith(base + os.sep):
            raise HTTPException(status_code=403,
                                detail="Path outside the figure directory.")
        if not os.path.isfile(target) or not target.lower().endswith(".png"):
            raise HTTPException(status_code=404, detail="Figure not found.")
        return FileResponse(target, media_type="image/png")
    except HTTPException:
        raise
    except Exception as e:
        raise _http_error(e, "/figure")


@app.post("/feedback")
def feedback(req: FeedbackRequest):
    """Log a thumbs up/down (and optional note) to a JSONL file for later review."""
    try:
        with open(FEEDBACK_LOG_PATH, "a") as f:
            f.write(json.dumps({"ts": time.time(), **req.model_dump()}) + "\n")
        return {"logged": True}
    except Exception as e:
        raise _http_error(e, "/feedback")


@app.post("/ingest")
def ingest_paper(req: IngestRequest):
    """Idempotent add. Fetch first (to normalise the id), then delete any existing
    chunks for that id BEFORE adding, so re-ingesting the same paper can never
    create duplicate embeddings. The on-disk index is upserted, not appended."""
    try:
        arxiv_id = req.arxiv_id.strip()
        if not ARXIV_ID_RE.match(arxiv_id):
            raise HTTPException(status_code=422,
                                detail="arxiv_id must look like 1706.03762")

        new_papers = fetch_papers([arxiv_id])
        if not new_papers:
            raise HTTPException(status_code=404,
                                detail=f"Paper {arxiv_id} could not be fetched from arXiv")

        paper_id = new_papers[0]["arxiv_id"]  # normalised (version suffix stripped)

        with _mutate_lock:
            existing = count_chunks(paper_id)
            if existing:
                get_vectorstore()._collection.delete(where={"arxiv_id": paper_id})
                logger.info("Re-ingest: removed %d existing chunk(s) for %s",
                            existing, paper_id)

            new_chunks = load_and_chunk(new_papers)
            if not new_chunks:
                raise HTTPException(status_code=422,
                                    detail=f"No extractable content found for {paper_id}")
            build_vectorstore(new_chunks, reset=False)

            index = [p for p in _load_index() if p["arxiv_id"] != paper_id]
            index.append({
                "arxiv_id": paper_id,
                "title": new_papers[0]["title"],
                "authors": new_papers[0]["authors"],
            })
            _save_index(index)

        return {
            "message": f"Ingested {len(new_chunks)} chunks from {new_papers[0]['title']}",
            "arxiv_id": paper_id,
            "chunks": len(new_chunks),
            "reingested": bool(existing),
            "total_chunks": count_chunks(),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise _http_error(e, "/ingest")


@app.delete("/papers/{arxiv_id}")
def delete_paper(arxiv_id: str):
    """Full cleanup with no orphans. Verify the paper exists (in Chroma or the
    index) BEFORE mutating anything, then delete: vector chunks, figure PNGs,
    figure cache entries, the PDF, and the index entry."""
    try:
        arxiv_id = arxiv_id.strip()
        if not ARXIV_ID_RE.match(arxiv_id):
            raise HTTPException(status_code=422,
                                detail="arxiv_id must look like 1706.03762")

        chunk_count = count_chunks(arxiv_id)
        index = _load_index()
        in_index = any(p["arxiv_id"] == arxiv_id for p in index)

        if chunk_count == 0 and not in_index:
            raise HTTPException(status_code=404,
                                detail=f"Paper {arxiv_id} not found (no chunks, not in index)")

        with _mutate_lock:
            # 1) vector chunks
            if chunk_count:
                get_vectorstore()._collection.delete(where={"arxiv_id": arxiv_id})
            remaining = count_chunks(arxiv_id)
            if remaining:
                # Defensive: should be 0. Surface it instead of silently leaving orphans.
                logger.warning("%d chunk(s) still present for %s after delete",
                               remaining, arxiv_id)

            # 2) extracted figure PNGs
            removed_images = 0
            for img_path in glob.glob(os.path.join(PDF_DIR, f"{arxiv_id}_p*_fig*.png")):
                try:
                    os.remove(img_path)
                    removed_images += 1
                except OSError:
                    pass

            # 3) figure caption/OCR cache entries
            cache = load_cache()
            stale_keys = [k for k in cache if k.startswith(f"{arxiv_id}_")]
            for k in stale_keys:
                del cache[k]
            if stale_keys:
                save_cache(cache)

            # 4) the downloaded PDF (so a re-ingest re-fetches fresh)
            pdf_path = os.path.join(PDF_DIR, f"{arxiv_id}.pdf")
            if os.path.exists(pdf_path):
                os.remove(pdf_path)

            # 5) the index entry
            _save_index([p for p in index if p["arxiv_id"] != arxiv_id])

        return {
            "message": f"Deleted paper {arxiv_id}",
            "removed_chunks": chunk_count,
            "removed_images": removed_images,
            "removed_cache_entries": len(stale_keys),
            "orphans_remaining": remaining,
            "total_chunks": count_chunks(),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise _http_error(e, "/delete")
