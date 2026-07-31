"""Every tunable in one place.

Paths are resolved relative to the project root (the directory containing this
package), not the current working directory, so `python scripts/ingest.py` and
`uvicorn app.main:app` both read and write the same ./data folder.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root before anything reads os.environ.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# --- Paths -------------------------------------------------------------------
# Everything generated at runtime lives under ./data so .gitignore is one line.
DATA_DIR = PROJECT_ROOT / "data"
PDF_DIR = str(DATA_DIR / "papers")
CHROMA_DIR = str(DATA_DIR / "chroma_db")
CACHE_PATH = str(DATA_DIR / "image_descriptions_cache.json")
PAPERS_INDEX_PATH = str(DATA_DIR / "papers_index.json")
FEEDBACK_LOG_PATH = str(DATA_DIR / "feedback.log")
TESSERACT_CMD = os.environ.get("TESSERACT_CMD", "")

DATA_DIR.mkdir(exist_ok=True)
Path(PDF_DIR).mkdir(exist_ok=True)

# --- Models ------------------------------------------------------------------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# FREE-TIER CHOICE: llama-3.1-8b-instant.
#   Free plan (per Groq docs, mid-2026):
#     llama-3.1-8b-instant   -> 30 RPM, 14,400 RPD, ~6K TPM,  500K tokens/DAY
#     llama-3.3-70b-versatile-> 30 RPM,  1,000 RPD, 12K TPM,  100K tokens/DAY
#   Each RAG answer costs ~2-4K tokens, so 8b's 500K/day budget = ~5x more questions
#   per day than 70b before you hit the wall. 8b is plenty for GROUNDED QA (the model
#   only has to synthesise retrieved chunks, not reason from scratch).
LLM_MODEL = os.environ.get("LLM_MODEL", "llama-3.1-8b-instant")
LLM_TEMPERATURE = 0.3
LLM_REGEN_TEMPERATURE = 0.7   # used ONLY when regenerating a disliked answer
LLM_MAX_TOKENS = 768          # concise research answers; protects the free-tier budget

# --- Chunking ----------------------------------------------------------------
# Sizes are measured in TOKENS (the splitter is built from the MiniLM tokenizer),
# not characters. all-MiniLM-L6-v2 hard-truncates at 256 tokens, so 240 keeps whole
# chunks inside the embedding window with headroom. Small chunks also mean fewer
# tokens per LLM call -> more questions/day under Groq's free-tier token budget.
# Changing this REQUIRES re-running ingestion.
CHUNK_SIZE = 240        # tokens (<= MiniLM's 256 hard cap)
CHUNK_OVERLAP = 40      # tokens

# --- Retrieval ---------------------------------------------------------------
RETRIEVER_K = 3              # final number of chunks sent to the LLM (keep small on free tier)
RETRIEVER_CANDIDATE_K = 12   # wider pool so several papers can be represented
MAX_CONTEXT_CHARS_PER_DOC = 1100   # ~240 tokens; trims any over-long chunk

# Cosine relevance from MiniLM runs low on this data (~0.2-0.5 for good matches),
# so keep the floor small. Retrieval is *guarded*: if the floor removes everything,
# we fall back to the top raw hits so a genuine query never returns empty.
RETRIEVER_MIN_RELEVANCE = 0.15
FIGURE_TABLE_FLOOR = 0.10    # even more lenient for explicit figure/table lookups
MAX_CHUNKS_PER_PAPER = 2     # cap per paper on multi-paper questions so one paper
                             # can't dominate the context window

# --- Table ingestion strategy (v15 rewrite) ----------------------------------
# pdfplumber's borderless table extraction SHREDS dense academic tables into fake
# columns ('T5-Sm | all'), so structured grid extraction is abandoned entirely.
# New approach: every 'Table N' caption anchors a RAW TEXT band (the PDF's native
# text layer already contains every cell value in reading order), and ONE cached
# Groq call per table reformats that raw text into a clean markdown grid at
# ingestion time. ~10-15 tables/paper -> trivially inside the Groq free tier,
# one-time cost, cached in image_descriptions_cache.json.
TABLE_REGION_ABOVE = 150     # pts of raw text captured above the caption line
TABLE_REGION_BELOW = 480     # pts captured below (arXiv captions usually sit above)
STRUCTURE_TABLES_WITH_LLM = True  # set False to store raw text only (zero LLM calls)
TABLE_LLM_MODEL = "llama-3.1-8b-instant"
TABLE_LLM_MAX_TOKENS = 700
TABLE_GRID_MAX_CHARS = 1800   # cap on the markdown grid stored per table chunk

# --- Image RAG policy --------------------------------------------------------
# Global policy: no per-paper hardcoding. Images become searchable text chunks
# during ingestion. Vision captioning is the ONLY GPU-accelerated step in the
# system, and it runs at ingestion time only -- never per query.
IMAGE_RAG_POLICY = {
    "enabled": True,
    "max_images_per_paper": 6,
    "min_width": 180,
    "min_height": 120,
    "render_dpi": 150,
    "use_ocr": True,
    "use_vision_caption": True,
    "caption_margin_above": 300,
    "caption_margin_below": 30,
}

# --- Corpus ------------------------------------------------------------------
PAPER_IDS = [
    "1706.03762",  # Attention Is All You Need
    "1810.04805",  # BERT
    "2005.11401",  # Retrieval-Augmented Generation (Lewis et al.)
    "1910.13461",  # BART
    "2004.04906",  # Dense Passage Retrieval (DPR)
    "1908.10084",  # Sentence-BERT - the architecture behind all-MiniLM-L6-v2
    "2004.12832",  # ColBERT - late-interaction dense retrieval
    "2007.01282",  # Fusion-in-Decoder - retrieval-augmented generation, FAISS-based
    "1910.10683",  # T5 - text-to-text transfer transformer
    "1804.07461",  # GLUE benchmark
    "2010.11929",  # Vision Transformer (ViT)
]

# --- Server ------------------------------------------------------------------
API_HOST = os.environ.get("API_HOST", "127.0.0.1")
API_PORT = int(os.environ.get("API_PORT", "8000"))
API_BASE_URL = os.environ.get("API_BASE_URL", f"http://{API_HOST}:{API_PORT}")
# CORS: locked to the local Streamlit origin by default. The notebook used "*"
# because it ran behind a throwaway Cloudflare tunnel; local dev doesn't need that.
CORS_ORIGINS = [
    o.strip() for o in os.environ.get(
        "CORS_ORIGINS", "http://localhost:8501,http://127.0.0.1:8501"
    ).split(",") if o.strip()
]
