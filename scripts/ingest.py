"""One-off ingestion: fetch the corpus, chunk it, embed it into Chroma.

    python scripts/ingest.py                # full corpus from config.PAPER_IDS, wipes the store
    python scripts/ingest.py --add 2010.11929   # add one paper, keep existing chunks

Runtime note: vision captioning is the slow part. With a CUDA GPU the full
11-paper run takes a few minutes; on CPU expect noticeably longer. Set
IMAGE_RAG_POLICY["use_vision_caption"] = False in config.py to skip it.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import PAPER_IDS, CHROMA_DIR, TABLE_LLM_MODEL
from app.ingestion.chunking import build_vectorstore, load_and_chunk, save_papers_index
from app.ingestion.download import fetch_papers


def preflight_groq_key():
    """Fail LOUD and EARLY if the Groq key is missing/dead, instead of
    discovering it table-by-table halfway through ingestion."""
    key = os.environ.get("GROQ_API_KEY", "")
    if not key:
        print("!! GROQ_API_KEY not set (check your .env). Ingestion will still run "
              "but tables store RAW text only.")
        return False
    try:
        from langchain_groq import ChatGroq
        ChatGroq(model=TABLE_LLM_MODEL, temperature=0.0, max_tokens=5).invoke("ping")
        print(f"Groq key OK (prefix {key[:4]}..., len {len(key)}).")
        return True
    except Exception as e:
        print(f"!! Groq key present but rejected: {e}\n"
              "   Generate a new key at console.groq.com. "
              "Tables will fall back to raw text this run.")
        return False


def report_device():
    try:
        import torch
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)} - vision captioning will use CUDA.")
        else:
            print("GPU: none detected - vision captioning falls back to CPU (slower).")
    except Exception:
        print("GPU: torch unavailable; skipping device check.")


def sanity_check(vectorstore):
    query = "What is self-attention?"
    results = vectorstore.similarity_search(query, k=3)
    print(f"\nTest query: '{query}'")
    for i, doc in enumerate(results, 1):
        print(f"\n  Result {i}")
        print(f"  Source: {doc.metadata.get('source', 'MISSING - check metadata')}")
        print(f"  Preview: {doc.page_content[:200]}...")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--add", metavar="ARXIV_ID", nargs="+",
                        help="add these papers instead of rebuilding the whole corpus")
    args = parser.parse_args()

    paper_ids = args.add or PAPER_IDS
    reset = args.add is None

    report_device()
    preflight_groq_key()

    print(f"\nStep 1/4: Fetching {len(paper_ids)} paper(s) from arXiv...")
    papers = fetch_papers(paper_ids)

    print("\nStep 2/4: Loading and chunking PDFs (text + tables + OCR/vision chunks)...")
    chunks = load_and_chunk(papers)
    print(f"\nTotal chunks created: {len(chunks)}")

    print(f"\nStep 3/4: Embedding and storing in ChromaDB (reset={reset})...")
    vectorstore = build_vectorstore(chunks, reset=reset)
    print(f"Stored at: {CHROMA_DIR}")

    print("\nStep 4/4: Sanity check...")
    sanity_check(vectorstore)

    if reset:
        save_papers_index(papers)


if __name__ == "__main__":
    main()
