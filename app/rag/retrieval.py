"""Retrieval and context selection.

The routing in select_context_docs() is the heart of the system. In order:
  0.   explicit "Table N" / "Figure N"  -> label lookup scoped to one paper
  0.5  summary intent                   -> deterministic early-page pull
  1.   general figure/table intent      -> content_type-filtered pass, paper-scoped
  2.   comparative intent               -> named-paper scoping, else round-robin
  3.   default                          -> stay on the named paper, reserve one
                                           slot for its opening pages

All of it is plain Python over the vector store -- no extra LLM calls.
"""

import re
from collections import Counter, OrderedDict, defaultdict

from langchain_core.documents import Document

from app.config import (
    FIGURE_TABLE_FLOOR,
    MAX_CHUNKS_PER_PAPER,
    MAX_CONTEXT_CHARS_PER_DOC,
    RETRIEVER_CANDIDATE_K,
    RETRIEVER_K,
    RETRIEVER_MIN_RELEVANCE,
    TABLE_GRID_MAX_CHARS,
)
from app.rag.store import get_vectorstore

# Keywords that suggest the user actually wants a table/figure, not just prose.
FIGURE_TABLE_HINTS = (
    "figure", "fig.", "table", "chart", "diagram", "graph", "plot", "image",
    "illustration", "architecture diagram", "shown in",
)

# Cues that a question spans/compares multiple papers.
MULTI_PAPER_HINTS = (
    "compare", "comparison", "versus", " vs ", "vs.", "difference between", "differ",
    "contrast", "both", "across", "each paper", "these papers", "all papers",
    "which paper", "trade-off", "tradeoff", "relative to", "compared to",
)


# Vague 'summarize / overview' queries don't semantically match any one chunk well
# (the word 'summarize' isn't in the abstract), so retrieval used to pull junk.
# Fix: detect summary intent and pull the target paper's EARLIEST pages directly
# (abstract + intro live there) instead of trusting pure similarity.
SUMMARY_HINTS = ("summarize", "summarise", "summary", "overview", "tl;dr", "tldr",
                 "what is this paper about", "what is the paper about", "about the paper",
                 "main contribution", "main idea", "key idea", "key contributions",
                 "in a nutshell", "briefly explain the paper", "abstract")

def wants_summary(question):
    q = (question or "").lower()
    return any(h in q for h in SUMMARY_HINTS)

def retrieve_early_pages(arxiv_id, max_docs=RETRIEVER_K):
    """Fetch the target paper's earliest text chunks straight from Chroma.
    Deterministic -- no similarity involved, so 'summarize X' always sees the
    abstract/intro. v14: single get() + python sort (avoids int-equality quirks
    in Chroma where-filters), then keep the lowest-page chunks."""
    docs = []
    try:
        got = get_vectorstore()._collection.get(
            where={"$and": [{"arxiv_id": arxiv_id}, {"content_type": "text"}]},
            include=["documents", "metadatas"],
        )
        pairs = list(zip(got.get("documents") or [], got.get("metadatas") or []))
        pairs.sort(key=lambda p: (p[1] or {}).get("page", 10**6))
        for txt, md in pairs[:max_docs]:
            docs.append(Document(page_content=txt, metadata=md or {}))
    except Exception as e:
        print(f"  retrieve_early_pages failed: {e}")
    return docs


def wants_figure_or_table(question):
    q = question.lower()
    return any(hint in q for hint in FIGURE_TABLE_HINTS)


def wants_multi_paper(question, scored_docs=None):
    """Multi-paper intent ONLY on an explicit comparative cue. The old
    'top candidates span >=3 papers' auto-trigger fired on ordinary single-topic
    follow-ups (e.g. 'explain it more elaborately') and scattered the context
    across unrelated papers -- that is what produced the incoherent 3-paper answer.
    Removed."""
    q = f" {question.lower()} "
    return any(hint in q for hint in MULTI_PAPER_HINTS)


def _dedupe(docs):
    seen, out = set(), []
    for d in docs:
        key = (d.metadata.get("arxiv_id"), d.metadata.get("page"),
               d.metadata.get("content_type", "text"), d.page_content[:80])
        if key not in seen:
            seen.add(key)
            out.append(d)
    return out


def retrieve_scored(query, k=RETRIEVER_CANDIDATE_K,
                    min_relevance=RETRIEVER_MIN_RELEVANCE, content_types=None):
    """Return [(doc, score)] sorted by relevance, above a lenient floor.

    Guarded: if the floor removes everything, fall back to the top raw hits so a
    genuine query never returns empty. `content_types` applies a Chroma metadata
    filter (used for the dedicated figure/table pass).
    """
    kwargs = {}
    if content_types:
        kwargs["filter"] = {"content_type": {"$in": list(content_types)}}
    try:
        scored = get_vectorstore().similarity_search_with_relevance_scores(query, k=k, **kwargs)
    except TypeError:
        # Older langchain-chroma used `where` instead of `filter`.
        if content_types:
            kwargs = {"where": {"content_type": {"$in": list(content_types)}}}
        scored = get_vectorstore().similarity_search_with_relevance_scores(query, k=k, **kwargs)
    scored.sort(key=lambda x: (x[1] is not None, x[1] or 0.0), reverse=True)
    kept = [(d, s) for d, s in scored if (s is None or s >= min_relevance)]
    if not kept and scored:
        kept = scored[:RETRIEVER_K]  # never starve a real query
    return kept


def diversify_by_paper(scored_docs, final_k=RETRIEVER_K,
                       max_per_paper=MAX_CHUNKS_PER_PAPER):
    """Round-robin across papers (best score first within each) so multi-paper
    questions get coverage instead of K chunks from a single paper."""
    buckets = OrderedDict()
    for d, _ in scored_docs:  # already sorted best-first
        buckets.setdefault(d.metadata.get("arxiv_id", "?"), []).append(d)
    selected, counts = [], defaultdict(int)
    while len(selected) < final_k:
        progressed = False
        for aid, docs in buckets.items():
            if counts[aid] < max_per_paper and counts[aid] < len(docs):
                selected.append(docs[counts[aid]])
                counts[aid] += 1
                progressed = True
                if len(selected) >= final_k:
                    break
        if not progressed:
            break
    return selected


EXPLICIT_LABEL_RE = re.compile(r"\b(fig(?:ure|\.)?|table)\s*\.?\s*(\d+)", re.IGNORECASE)

def parse_explicit_label(question):
    """Return a normalised 'Table N' / 'Figure N' if the question names one."""
    m = EXPLICIT_LABEL_RE.search(question or "")
    if not m:
        return None
    kind = "Figure" if m.group(1).lower().startswith("fig") else "Table"
    return f"{kind} {m.group(2)}"

PAPER_ALIASES = {
    # NOTE: put the MOST specific aliases first; matching tries them in order.
    # Include the short natural names people actually type ("attention paper",
    # not just the full title) -- missing these was why "Table 2 of the Attention
    # paper" fell through to the semantic vote and returned another paper's Table 2.
    "1706.03762": ["attention is all you need","attention paper","transformer paper",
                   "self-attention paper","attention"],
    "1810.04805": ["bert paper","bert"],
    "2005.11401": ["retrieval-augmented generation","rag paper","lewis et al","rag"],
    "1910.13461": ["bart paper","bart"],
    "2004.04906": ["dense passage retrieval","dpr paper","dpr"],
    "1908.10084": ["sentence-bert","sbert","sentence bert","sentence transformer"],
    "2004.12832": ["colbert paper","colbert"],
    "2007.01282": ["fusion-in-decoder","fusion in decoder","fid paper","fid"],
    "1910.10683": ["exploring the limits of transfer learning","text-to-text",
                   "t5 paper","t5"],
    "1804.07461": ["glue benchmark","glue paper","glue"],
    "2010.11929": ["vision transformer","image is worth","16x16","vit"],
}

def resolve_target_paper(question, scored):
    """Deterministic paper scoping: explicit arXiv id > title/alias match > semantic
    majority. Replaces infer_target_paper, whose semantic-only vote picked the WRONG
    paper on title-named table queries (every paper has a 'Table N')."""
    q = " ".join((question or "").lower().replace("-", " ").split())
    m = re.search(r"\b(\d{4}\.\d{4,5})\b", q)
    if m:
        return m.group(1)
    for aid, names in PAPER_ALIASES.items():
        for n in names:
            n_norm = " ".join(n.replace("-", " ").split())
            if re.search(r"\b" + re.escape(n_norm) + r"\b", q):
                return aid
    # Semantic fallback is UNRELIABLE for "Table N in <paper>" queries (every paper
    # has a Table N), so only trust it when the top hits AGREE strongly. Otherwise
    # return None and let the caller stay honest instead of guessing a wrong paper.
    top = [d.metadata.get("arxiv_id") for d, _ in scored[:5] if d.metadata.get("arxiv_id")]
    if not top:
        return None
    winner, count = Counter(top).most_common(1)[0]
    has_label = bool(re.search(r"\b(table|figure)\s+\d+\b", q))
    if has_label and count < 3:
        # weak agreement on a labelled query -> don't guess
        return None
    return winner

_FIG_WORDS = ("word cloud","network diagram","block diagram","architecture diagram",
              "flowchart","interconnected nodes","this image is a chart",
              "this is a figure caption")
def _is_figure_like_table(doc):
    """Distrust ONLY unlabeled 'table' chunks that read like figures (the real junk).
    Chunks with a genuine 'Table N' label are trusted even if MoonDream's prose waffled."""
    if doc.metadata.get("content_type") != "table":
        return False
    if doc.metadata.get("table_label"):
        return False
    t = (doc.page_content or "").lower()
    return any(w in t for w in _FIG_WORDS)

def retrieve_by_label(query, label, arxiv_id=None, k=RETRIEVER_CANDIDATE_K):
    """Deterministic lookup for a specific 'Table N' / 'Figure N', SCOPED to one paper.

    Without the arxiv_id scope this returned 'Table 2' from whatever paper matched
    best -- i.e. the wrong paper -- which is exactly what caused the bad table answers.
    We now filter on (arxiv_id AND label). If the named paper has no such labelled
    chunk we return [] (so the caller falls through to normal retrieval) rather than
    handing back another paper's table.
    """
    field = "table_label" if label.startswith("Table") else "figure_label"
    if arxiv_id:
        flt = {"$and": [{"arxiv_id": arxiv_id}, {field: label}]}
    else:
        flt = {field: label}
    hits = []
    try:
        docs = get_vectorstore().similarity_search(query, k=k, filter=flt)
        hits = [d for d in docs if d.metadata.get(field) == label
                and (arxiv_id is None or d.metadata.get("arxiv_id") == arxiv_id)]
    except Exception:
        hits = []
    if not hits:  # fallback text-scan, still scoped to the target paper
        ctypes = {"table"} if label.startswith("Table") else {"figure"}
        pool = retrieve_scored(query, k=k, min_relevance=0.0, content_types=ctypes)
        lab = label.lower()
        hits = [d for d, _ in pool
                if (arxiv_id is None or d.metadata.get("arxiv_id") == arxiv_id)
                and (d.metadata.get(field, "").lower() == lab or lab in d.page_content.lower())]
    return hits

def resolve_named_papers(question):
    """Every paper explicitly named in the question, in the order mentioned.

    resolve_target_paper() returns a single paper, which is right for "Table 2 of
    the BERT paper" but wrong for "compare BERT and BART". Aliases are tried
    LONGEST FIRST so "sentence-bert" wins over the bare "bert" alias, and each
    matched span is blanked out so one phrase can't resolve to two papers.
    """
    q = " ".join((question or "").lower().replace("-", " ").split())
    pairs = []
    for aid, names in PAPER_ALIASES.items():
        for n in names:
            pairs.append((aid, " ".join(n.replace("-", " ").split())))
    pairs.sort(key=lambda p: -len(p[1]))          # longest alias first

    found = []
    for aid, alias in pairs:
        if aid in found:
            continue
        m = re.search(r"\b" + re.escape(alias) + r"\b", q)
        if m:
            found.append(aid)
            q = q[:m.start()] + " " * (m.end() - m.start()) + q[m.end():]
    return found


def select_context_docs(question, final_k=RETRIEVER_K):
    scored = retrieve_scored(question)
    text_scored  = [(d, s) for d, s in scored if d.metadata.get("content_type", "text") == "text"]
    other_scored = [(d, s) for d, s in scored if d.metadata.get("content_type", "text") != "text"]

    target = resolve_target_paper(question, scored)   # title/id first, semantic last

    # 0) explicit "Table N" / "Figure N" -> scoped label lookup; honest fallback if the
    #    named paper lacks it (e.g. GLUE has no Table 1/2) -- never another paper's.
    label = parse_explicit_label(question)
    if label:
        labelled = retrieve_by_label(question, label, arxiv_id=target) if target else []
        if labelled:
            same_text = [d for d, _ in text_scored if d.metadata.get("arxiv_id") == target]
            return _dedupe(labelled + same_text)[:final_k], scored
        if target:  # target paper has no such Table/Figure N -> stay on it (or fall back)
            same_text = [d for d, _ in text_scored if d.metadata.get("arxiv_id") == target]
            return _dedupe(same_text)[:final_k], scored

    # 0.5) summary/overview intent -> deterministic early-page pull for the target
    #      paper (abstract + intro), topped up with the best same-paper text hits.
    if wants_summary(question) and target:
        early = retrieve_early_pages(target, max_docs=final_k)
        same_text = [d for d, _ in text_scored if d.metadata.get("arxiv_id") == target]
        docs = _dedupe(early + same_text)[:final_k]
        print(f"  [summary-intent] target={target} early={len(early)} same_text={len(same_text)}")
        if docs:
            return docs, scored

    # 1) general figure/table intent -> SCOPE to the target paper (fixes T5 #4)
    if wants_figure_or_table(question):
        ft = retrieve_scored(question, content_types={"figure", "table"},
                             min_relevance=FIGURE_TABLE_FLOOR)
        if target:
            ft_scoped = [(d, s) for d, s in ft if d.metadata.get("arxiv_id") == target]
            clean = [(d, s) for d, s in ft_scoped if not _is_figure_like_table(d)]
            ft_use = clean or ft_scoped   # don't starve if every candidate was flagged
            txt_t = [d for d, _ in text_scored if d.metadata.get("arxiv_id") == target]
            return _dedupe([d for d, _ in ft_use] + txt_t)[:final_k], scored
        docs = _dedupe([d for d, _ in ft] + [d for d, _ in text_scored])[:final_k]
        return docs, scored

    # 2) multi-paper / comparative
    if wants_multi_paper(question, scored):
        # If the question NAMES papers ("compare BERT and BART"), retrieve from
        # exactly those papers. Similarity alone picks title look-alikes --
        # "BERT" scores well against Sentence-BERT and ColBERT, which crowded
        # the real BERT paper out of the results and led to ColBERT's numbers
        # being reported as BART's.
        named = resolve_named_papers(question)
        if len(named) >= 2:
            per_paper = max(1, final_k // len(named))
            picked = []
            for aid in named:
                same = [d for d, _ in scored if d.metadata.get("arxiv_id") == aid]
                if not same:
                    # No chunk retrieved for a paper the user explicitly named:
                    # pull its early pages so the comparison isn't one-sided.
                    same = retrieve_early_pages(aid, max_docs=per_paper)
                picked.extend(same[:per_paper])
            missing = [a for a in named
                       if not any(d.metadata.get("arxiv_id") == a for d in picked)]
            if missing:
                print(f"  [compare] no chunks found for: {missing}")
            print(f"  [compare] named papers: {named} -> {len(picked)} chunks")
            if picked:
                return _dedupe(picked)[:final_k], scored

        base = text_scored if len(text_scored) >= final_k else scored
        return _dedupe(diversify_by_paper(base, final_k=final_k))[:final_k], scored

    # 3) default: if the question names a paper, STAY on that paper and make sure
    #    its opening pages are represented.
    #    Bug this fixes: this branch used to ignore `target` entirely and rank
    #    purely by similarity, so "what is BERT?" returned BERT's appendix pages
    #    (p.13/p.16) instead of the abstract. Those pages mention BERT only in
    #    passing, so the passing-mention rule in qa_prompt then answered
    #    "BERT is not among the indexed papers" -- about an indexed paper that
    #    retrieval had correctly found.
    #    Note: a plain "top up if short" does NOT fix this -- the same-paper hits
    #    already fill final_k. One slot is RESERVED for early pages instead,
    #    because similarity ranks keyword-dense pages (ablations, appendices)
    #    above the abstract, which is where a paper actually defines itself.
    if target:
        same_text = [d for d, _ in text_scored if d.metadata.get("arxiv_id") == target]
        early = retrieve_early_pages(target, max_docs=1)
        docs = _dedupe(early + same_text)[:final_k]
        if len(docs) < final_k:
            docs = _dedupe(docs + [d for d, _ in text_scored]
                                + [d for d, _ in other_scored])[:final_k]
        if docs:
            return docs, scored

    docs = [d for d, _ in text_scored][:final_k]
    if len(docs) < final_k:
        docs = _dedupe(docs + [d for d, _ in other_scored])[:final_k]
    return docs, scored

def trim_text(text, max_chars=MAX_CONTEXT_CHARS_PER_DOC, preserve_lines=False):
    if preserve_lines:
        # table grids need their newlines (markdown rows); trim by length only
        text = (text or "").strip()
        if len(text) <= max_chars:
            return text
        return text[:max_chars].rsplit("\n", 1)[0] + "\n... (truncated)"
    text = " ".join((text or "").split())
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + " ..."


def format_docs(docs):
    labels = {"figure": "[FIGURE]", "table": "[TABLE]", "text": "[TEXT]"}
    formatted = []
    for doc in docs:
        content_type = doc.metadata.get("content_type", "text")
        source = doc.metadata.get("source", "unknown")
        if content_type == "table":
            # allow the full stored grid (already capped at ingest by
            # TABLE_GRID_MAX_CHARS) -- the whole point is the LLM sees real values
            content = trim_text(doc.page_content,
                                max_chars=TABLE_GRID_MAX_CHARS + 400,
                                preserve_lines=True)
        else:
            content = trim_text(doc.page_content)
        formatted.append(f"SOURCE: {source}\n{labels.get(content_type, '[TEXT]')} {content}")
    return "\n\n".join(formatted)
