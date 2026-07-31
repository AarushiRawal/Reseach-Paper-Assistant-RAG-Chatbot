"""On-disk cache for the expensive, deterministic parts of ingestion.

Two things get cached here, both keyed per paper:
  * MoonDream vision captions + OCR text for figure crops
  * the LLM-reformatted markdown grid for each table

Both are expensive (GPU time / a Groq call) and never change for a given PDF,
so a re-ingest of an unchanged paper costs almost nothing.
"""

import json
import os

from app.config import CACHE_PATH


def load_cache():
    if os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH) as f:
                content = f.read().strip()
                if not content:
                    return {}
                return json.loads(content)
        except json.JSONDecodeError:
            print(f"Warning: {CACHE_PATH} was corrupted/unreadable, starting fresh.")
            return {}
    return {}


def save_cache(cache):
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)
