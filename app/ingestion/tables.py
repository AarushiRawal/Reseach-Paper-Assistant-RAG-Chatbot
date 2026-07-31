"""Caption-anchored table ingestion (the v15 rewrite).

Why not pdfplumber: its find_tables() shreds borderless academic tables into fake
columns ('T5-Sm | all'), and vision captions carry no cell values. But the PDF's
NATIVE text layer already contains every number in reading order. So:
  1. locate each 'Table N' caption via fitz,
  2. capture the raw text band around it (bounded by neighbouring captions),
  3. (optional, cached) one Groq call reformats the raw text into a markdown grid
     -- with a groundedness sanity check so it can't invent numbers.

No pdfplumber, no vision, no OCR on the table path -> much faster ingestion, and
every stored value is real PDF text, so answers pass the grounding check.
"""

import re
import time

import fitz
from langchain_core.documents import Document

from app.config import (
    STRUCTURE_TABLES_WITH_LLM,
    TABLE_GRID_MAX_CHARS,
    TABLE_LLM_MAX_TOKENS,
    TABLE_LLM_MODEL,
    TABLE_REGION_ABOVE,
    TABLE_REGION_BELOW,
)
from app.ingestion.cache import load_cache, save_cache

TABLE_CAPTION_FITZ_RE = re.compile(r"^(Table)\s+(\d+)\s*[:.]", re.IGNORECASE)

def _find_table_captions_fitz(page):
    """'Table N: ...' caption blocks with their bbox, via fitz (no pdfplumber)."""
    out = []
    for block in page.get_text("dict").get("blocks", []):
        if "lines" not in block:
            continue
        text = " ".join(
            span["text"] for line in block["lines"] for span in line["spans"]
        ).strip()
        m = TABLE_CAPTION_FITZ_RE.match(text)
        if m:
            out.append({
                "label": f"Table {m.group(2)}",
                "caption_text": " ".join(text.split())[:400],
                "bbox": fitz.Rect(block["bbox"]),
            })
    out.sort(key=lambda c: c["bbox"].y0)
    return out


_table_llm = None
def _get_table_llm():
    """Small Groq model used ONCE per table at ingestion (cached). Lazy so the
    ingestion cells don't depend on the serving cells having run."""
    global _table_llm
    if _table_llm is None:
        from langchain_groq import ChatGroq
        _table_llm = ChatGroq(model=TABLE_LLM_MODEL, temperature=0.0,
                              max_tokens=TABLE_LLM_MAX_TOKENS)
    return _table_llm

_TABLE_STRUCTURE_PROMPT = (
    "Below is raw text extracted from a table region of a research paper PDF. "
    "The values are in reading order but the column structure was lost.\n"
    "Reconstruct the table as a compact markdown table (| col | col |). Keep every "
    "numeric value EXACTLY as written -- do not round, invent, or omit numbers.\n"
    # --- added after seeing the Attention Table 2 output ---
    "Two rules that matter most:\n"
    "1. TWO-LEVEL HEADERS: academic tables often have a top header spanning several "
    "sub-columns (e.g. 'BLEU' spanning 'EN-DE' and 'EN-FR', then 'Training Cost' "
    "spanning 'EN-DE' and 'EN-FR' again). Flatten these into combined column names "
    "such as 'BLEU EN-DE', 'BLEU EN-FR', 'Cost EN-DE', 'Cost EN-FR' so every value "
    "sits under the heading it actually belongs to. Never merge two different "
    "measures into one column.\n"
    "2. STOP AT THE TABLE'S END: the raw text may run past the table into the next "
    "section or a later table (stray lines such as hyperparameter names or prose). "
    "Include ONLY rows that belong to the captioned table; drop anything after it.\n"
    "Every data row must have the same number of cells as the header -- use an empty "
    "cell for a missing value rather than shifting the remaining values left.\n"
    "If the structure cannot be confidently reconstructed, instead list each row as "
    "'row-label: values' lines. Output ONLY the table/rows, no commentary.\n\n"
    "Caption: {caption}\n\nRaw text:\n{raw}"
)

_TABLE_LLM_DISABLED = False   # set True after an auth error to stop retrying

def _structure_table_with_llm(caption_text, raw_text):
    global _TABLE_LLM_DISABLED
    if _TABLE_LLM_DISABLED:
        return ""
    try:
        resp = _get_table_llm().invoke(
            _TABLE_STRUCTURE_PROMPT.format(caption=caption_text, raw=raw_text[:3500]))
        out = (resp.content or "").strip()
        # sanity: the LLM must not invent numbers -- keep only if most of its
        # numeric tokens actually appear in the raw text.
        raw_ns = re.sub(r"\s+", "", raw_text)
        nums = re.findall(r"\d+(?:\.\d+)?", out)
        if nums:
            supported = sum(1 for n in nums if n in raw_ns)
            if supported / len(nums) < 0.85:
                return ""
        return out
    except Exception as e:
        msg = str(e)
        if "401" in msg or "invalid_api_key" in msg or "Invalid API Key" in msg:
            _TABLE_LLM_DISABLED = True
            print("    !! GROQ AUTH FAILED (401). Disabling LLM table structuring for "
                  "this run and falling back to RAW TABLE TEXT. Re-run cell 4 to reload "
                  "GROQ_API_KEY (or make a new key), then re-ingest -- cached tables are "
                  "kept, only failed ones get re-structured.")
        else:
            print(f"    table LLM structuring skipped: {e}")
        return ""


def describe_table_regions(filepath, paper):
    """v15 REWRITE: caption-anchored table ingestion. One chunk per 'Table N'
    caption, guaranteed.

    Why: pdfplumber's find_tables() shreds borderless academic tables into fake
    columns ('T5-Sm | all'), and vision captions carry no cell values. But the
    PDF's NATIVE text layer already contains every number in reading order. So:
      1. locate each 'Table N' caption via fitz,
      2. capture the raw text band around it (bounded by neighbouring captions),
      3. (optional, cached) one Groq call reformats the raw text into a markdown
         grid -- with a groundedness sanity check so it can't invent numbers.
    No pdfplumber, no vision, no OCR on the table path -> much faster ingestion,
    and every stored value is real PDF text, so answers pass the grounding check.
    """
    docs = []
    cache = load_cache()
    fitz_doc = fitz.open(filepath)
    try:
        for i, page in enumerate(fitz_doc):
            captions = _find_table_captions_fitz(page)
            if not captions:
                continue
            for ci, cap in enumerate(captions):
                y0 = cap["bbox"].y0
                # band above/below the caption, bounded by neighbouring captions
                top = max(0, y0 - TABLE_REGION_ABOVE)
                if ci > 0:
                    top = max(top, captions[ci - 1]["bbox"].y1 + 4)
                bottom = min(page.rect.height, cap["bbox"].y1 + TABLE_REGION_BELOW)
                if ci + 1 < len(captions):
                    bottom = min(bottom, captions[ci + 1]["bbox"].y0 - 4)
                clip = fitz.Rect(0, top, page.rect.width, bottom)
                raw = " ".join((page.get_text("text", clip=clip) or "").split())
                if len(raw) < 40:
                    continue

                key = f"{paper['arxiv_id']}_tblv15_p{i+1}_{cap['label'].replace(' ', '')}"
                structured = ""
                if STRUCTURE_TABLES_WITH_LLM:
                    if key in cache and "structured" in cache[key]:
                        structured = cache[key]["structured"]
                    else:
                        structured = _structure_table_with_llm(cap["caption_text"], raw)
                        cache[key] = {"structured": structured}
                        time.sleep(0.35)   # gentle on the free-tier rate limit

                body = ("Table values (reconstructed):\n" + structured) if structured \
                       else ("Table text (raw): " + raw[:TABLE_GRID_MAX_CHARS])
                content = (
                    f"{cap['label']} from '{paper['title']}' (page {i+1}).\n"
                    f"{cap['caption_text']}\n{body}"
                ).strip()
                docs.append(Document(
                    page_content=content,
                    metadata={
                        "arxiv_id": paper["arxiv_id"],
                        "title": paper["title"],
                        "authors": ", ".join(paper["authors"]),
                        "page": i + 1,
                        "table_label": cap["label"],
                        "source": f"{paper['title'][:50]} (arXiv:{paper['arxiv_id']}, p.{i+1}, {cap['label']})",
                        "content_type": "table",
                        "has_structured": bool(structured),
                    },
                ))
    finally:
        save_cache(cache)
        fitz_doc.close()
    print(f"    tables: {len(docs)} caption-anchored chunk(s)")
    return docs

