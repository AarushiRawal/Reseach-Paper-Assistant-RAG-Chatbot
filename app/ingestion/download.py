"""arXiv metadata lookup + resilient PDF download."""

import os
import time

import arxiv
import requests

from app.config import PDF_DIR

def _download_pdf(pdf_url, filepath, retries=4, connect_to=15, read_to=120):
    """Stream a PDF to disk with a SEPARATE connect vs read timeout and retries.
    The old code passed timeout=30 (applies to BOTH), so a slow arXiv read on a
    big PDF (e.g. T5, 60+ pages) raised ReadTimeout with no retry. Streaming +
    a 120s read timeout + backoff fixes the 'read timeout when adding a paper' bug.
    """
    last = None
    for attempt in range(1, retries + 1):
        try:
            with requests.get(pdf_url, stream=True,
                              timeout=(connect_to, read_to)) as r:
                r.raise_for_status()
                with open(filepath, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 16):
                        if chunk:
                            f.write(chunk)
            if os.path.getsize(filepath) < 1024:
                raise IOError("downloaded file suspiciously small")
            return filepath
        except Exception as e:
            last = e
            wait = min(2 ** attempt, 20)
            print(f"    pdf download failed ({e}); retry {attempt}/{retries} in {wait}s...")
            if os.path.exists(filepath):
                try: os.remove(filepath)
                except OSError: pass
            time.sleep(wait)
    raise RuntimeError(f"could not download {pdf_url} after {retries} attempts: {last}")


def fetch_papers(arxiv_ids, max_retries=5):
    """Download PDFs and collect metadata for each paper.
    Retries with backoff on arXiv 429/503 (rate limiting) instead of
    failing the whole batch outright.
    """
    # Longer delay between arXiv API calls + more built-in retries --
    # the default settings are too aggressive for an 11-paper batch
    # and trip arXiv's rate limiter (429) which can cascade into a 503.
    client = arxiv.Client(page_size=100, delay_seconds=5, num_retries=5)
    search = arxiv.Search(id_list=arxiv_ids)

    for attempt in range(1, max_retries + 1):
        try:
            results = list(client.results(search))
            break
        except Exception as e:
            if attempt == max_retries:
                raise
            wait = 15 * attempt
            print(f"arXiv request failed ({e}); retrying in {wait}s "                  f"(attempt {attempt}/{max_retries})...")
            time.sleep(wait)

    papers = []
    for result in results:
        arxiv_id = result.get_short_id().split("v")[0]
        filename = f"{arxiv_id}.pdf"
        filepath = os.path.join(PDF_DIR, filename)
        if not os.path.exists(filepath):
            pdf_url = next((l.href for l in result.links if l.title == "pdf"), None)
            if pdf_url is None:
                pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            _download_pdf(pdf_url, filepath)
            time.sleep(1)
        papers.append({
            "arxiv_id": arxiv_id,
            "title": result.title,
            "authors": [a.name for a in result.authors[:3]],
            "filepath": filepath,
        })
        print(f"Fetched: {result.title}")
    return papers