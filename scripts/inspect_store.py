"""Inspect what is actually in the vector store -- the debugging tool that found
the mislabelled table/figure chunks and the missing-paper retrieval bugs.

    python scripts/inspect_store.py
    python scripts/inspect_store.py --paper 1910.10683
"""

import argparse
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.rag.retrieval import PAPER_ALIASES
from app.rag.store import get_vectorstore


def looks_like_figure(text):
    t = (text or "").lower()
    return any(w in t for w in ("diagram", "architecture", "arrow", "flowchart",
                                "block diagram", "screenshot")) and \
        not re.search(r"\d+(\.\d+)?\s*(%|bleu|f1|accuracy|score)", t)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper", help="restrict output to one arxiv_id")
    args = parser.parse_args()

    got = get_vectorstore()._collection.get(include=["metadatas", "documents"])
    metas, docs = got["metadatas"], got["documents"]
    print(f"TOTAL chunks in store: {len(metas)}\n")

    targets = ({args.paper: args.paper} if args.paper
               else {aliases[0]: aid for aid, aliases in PAPER_ALIASES.items()})

    for name, aid in targets.items():
        idx = [i for i, m in enumerate(metas) if m.get("arxiv_id") == aid]
        if not idx:
            continue
        ctypes = Counter(metas[i].get("content_type", "text") for i in idx)
        print(f"=== {name} ({aid}) : {len(idx)} chunks {dict(ctypes)} ===")
        for i in idx:
            m = metas[i]
            if m.get("content_type") in ("table", "figure"):
                lab = m.get("table_label") or m.get("figure_label") or "(no label)"
                flag = ("  <-- reads like a FIGURE"
                        if m.get("content_type") == "table" and looks_like_figure(docs[i])
                        else "")
                preview = re.sub(r"\s+", " ", docs[i])[:130]
                print(f"  [{m['content_type']:6}] label={lab!r:14} p{m.get('page')} "
                      f":: {preview}{flag}")
        print()

    print("=== Table 1 / Table 2 coverage across corpus ===")
    for want in ("Table 1", "Table 2"):
        have = sorted({m.get("arxiv_id") for m in metas if m.get("table_label") == want})
        print(f"  {want}: present for arxiv_ids -> {have}")


if __name__ == "__main__":
    main()
