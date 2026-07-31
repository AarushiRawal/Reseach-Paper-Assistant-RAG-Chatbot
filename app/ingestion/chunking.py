"""Token-aware chunking + vector store construction.

Chunk sizes are measured in TOKENS using MiniLM's own tokenizer so every chunk
fits inside the 256-token embedding window instead of being silently truncated.
"""

import json
import os
import shutil

import fitz
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

from app.config import (
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    PAPERS_INDEX_PATH,
)
from app.ingestion.figures import extract_and_describe_figures
from app.ingestion.tables import describe_table_regions

_hf_tokenizer = None
def _get_chunk_tokenizer():
    """MiniLM's own tokenizer, loaded once. Local + free. Lets us size chunks by
    TOKENS so they always fit the 256-token embedding window."""
    global _hf_tokenizer
    if _hf_tokenizer is None:
        _hf_tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
    return _hf_tokenizer


def load_pdf_pages_with_fitz(filepath):
    """Load one Document per PDF page without importing PyPDFLoader/langchain-community."""
    pages = []
    with fitz.open(filepath) as pdf:
        total_pages = pdf.page_count
        for page_index, page in enumerate(pdf):
            text = page.get_text("text") or ""
            pages.append(Document(
                page_content=text,
                metadata={
                    "source_file": filepath,
                    "page": page_index,
                    "total_pages": total_pages,
                },
            ))
    return pages


def load_and_chunk(papers):
    # Task 3: structure-first, TOKEN-sized chunking (fits MiniLM's 256-token window,
    # keeps per-call tokens low for the free tier).
    splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        _get_chunk_tokenizer(),
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""],
        keep_separator=True,
    )
    all_chunks = []
    for paper in papers:
        pages = load_pdf_pages_with_fitz(paper["filepath"])
        chunks = splitter.split_documents(pages)
        for chunk in chunks:
            page_number = chunk.metadata.get("page", 0) + 1
            chunk.metadata.update({
                "arxiv_id": paper["arxiv_id"],
                "title": paper["title"],
                "authors": ", ".join(paper["authors"]),
                "page": page_number,
                "source": f"{paper['title'][:50]} (arXiv:{paper['arxiv_id']}, p.{page_number})",
                "content_type": "text",
            })
        all_chunks.extend(chunks)

        # Tables via the VISION path (render region -> MoonDream + OCR), not pdfplumber's
        # markdown grid, which mangles dense academic tables. Keeps the 'Table N' label.
        table_docs = describe_table_regions(paper["filepath"], paper)
        all_chunks.extend(table_docs)
        table_count = len(table_docs)

        image_docs = extract_and_describe_figures(paper["filepath"], paper)
        all_chunks.extend(image_docs)

        print(
            f"  {paper['title'][:50]}... -> "
            f"{len(chunks)} text + {table_count} tables + {len(image_docs)} image-derived chunks"
        )
    return all_chunks
def _dedupe_chunks(chunks):
    """Drop exact-duplicate chunks (same paper+page+type+opening text) before they
    ever reach the embedder. Cheap insurance against double-ingestion."""
    seen, out = set(), []
    for d in chunks:
        key = (d.metadata.get("arxiv_id"), d.metadata.get("page"),
               d.metadata.get("content_type", "text"), d.page_content[:120])
        if key not in seen:
            seen.add(key)
            out.append(d)
    dropped = len(chunks) - len(out)
    if dropped:
        print(f"  _dedupe_chunks: dropped {dropped} duplicate chunk(s) before embedding")
    return out

def _sanitize_metadata(chunks):
    """Chroma only accepts str/int/float/bool metadata values. Anything else
    (notably `authors`, which is a list) is rejected at upsert time with a
    ValueError. Coerce at the boundary so no upstream site has to remember.

    Lists become comma-joined strings, which keeps them searchable by the
    grounding check; None is dropped; anything else is str()'d.
    """
    fixed = 0
    for doc in chunks:
        for key, value in list(doc.metadata.items()):
            if isinstance(value, (str, int, float, bool)):
                continue
            if value is None:
                del doc.metadata[key]
            elif isinstance(value, (list, tuple, set)):
                doc.metadata[key] = ", ".join(str(v) for v in value)
            else:
                doc.metadata[key] = str(value)
            fixed += 1
    if fixed:
        print(f"  _sanitize_metadata: coerced {fixed} non-scalar metadata value(s)")
    return chunks

def build_vectorstore(chunks, reset=False):
    """FIX (was silently duplicating on re-run): reset=True used to call
    Chroma.from_documents() against an EXISTING persist_directory, which does not
    clear old data -- it just adds on top. If a prior run crashed partway through
    (e.g. the 'readonly database' error) any chunks it had already written stayed,
    and the next successful run added a full second copy on top of them.
    Now reset=True EXPLICITLY drops the collection (and the on-disk folder as a
    belt-and-suspenders fallback) before writing, so a reset is always a true wipe."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    chunks = _dedupe_chunks(chunks)
    chunks = _sanitize_metadata(chunks)
    if reset:
        try:
            import chromadb
            chromadb.api.client.SharedSystemClient.clear_system_cache()
        except Exception:
            pass
        shutil.rmtree(CHROMA_DIR, ignore_errors=True)

    if reset or not os.path.exists(CHROMA_DIR):
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=CHROMA_DIR,
            collection_name="arxiv_papers",
            collection_metadata={"hnsw:space": "cosine"},
        )
    else:
        vectorstore = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=embeddings,
            collection_name="arxiv_papers",
            collection_metadata={"hnsw:space": "cosine"},
        )
        vectorstore.add_documents(chunks)
    return vectorstore

def save_papers_index(papers):
    index = [{"arxiv_id": p["arxiv_id"], "title": p["title"], "authors": p["authors"]}
             for p in papers]
    with open(PAPERS_INDEX_PATH, "w") as f:
        json.dump(index, f, indent=2)
    print(f"Saved index with {len(index)} papers")
