"""Lazy singletons for the embedding model, the Chroma store, and the Groq LLMs.

In the notebook these were module-level globals created at import time. That is
fine in one long-lived process but wrong for a package: importing anything from
`app.rag` would download MiniLM and open Chroma, which makes `scripts/ingest.py`,
the tests, and even `--help` slow. Everything here is created on first use instead.
"""

from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import (
    CHROMA_DIR,
    EMBEDDING_MODEL,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_REGEN_TEMPERATURE,
    LLM_TEMPERATURE,
)

_embeddings = None
_vectorstore = None
_llm = None
_llm_regen = None


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _embeddings


def get_vectorstore():
    """The persisted Chroma collection. Cosine space, so relevance scores are
    bounded and comparable across queries (retrieval thresholds depend on this)."""
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=get_embeddings(),
            collection_name="arxiv_papers",
            collection_metadata={"hnsw:space": "cosine"},
        )
    return _vectorstore


def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatGroq(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
        )
    return _llm


def get_llm_regen():
    """Warmer variant, used ONLY when the user asks to regenerate a disliked answer."""
    global _llm_regen
    if _llm_regen is None:
        _llm_regen = ChatGroq(
            model=LLM_MODEL,
            temperature=LLM_REGEN_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
        )
    return _llm_regen


def count_chunks(arxiv_id=None):
    """Total chunks, or chunks belonging to one paper."""
    collection = get_vectorstore()._collection
    if arxiv_id is None:
        return collection.count()
    got = collection.get(where={"arxiv_id": arxiv_id})
    return len(got["ids"]) if got and got.get("ids") else 0
