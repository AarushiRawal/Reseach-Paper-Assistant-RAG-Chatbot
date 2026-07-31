"""Programmatic groundedness verification -- model-agnostic and essentially free.

Every 'risky' numeric and every proper name in a generated answer is checked
against the retrieved context. Anything unsupported triggers a rewrite pass; if
the rewrite still fails, we fall back rather than serve an ungrounded number.

Deliberately NOT an LLM judge: a second model call would double token spend and
introduce its own hallucinations. Regex atom-matching cannot catch paraphrased
falsehoods, which is the known limitation of this approach.
"""

import re

from app.config import TABLE_GRID_MAX_CHARS
from app.rag.retrieval import parse_explicit_label, trim_text

_NUM_RE  = re.compile(
    r'\d(?:[\d,]*\d)?(?:\.\d+)?(?:\s?%|\s(?:billion|million|thousand)|[BMK])?', re.I)
_NAME_RE = re.compile(r'([A-Z][a-z]+(?:\s+et\s+al\.?|(?:\s+[A-Z][a-z]+){1,3}))')

def _gnorm(s): return re.sub(r'\s+', ' ', (s or '')).lower()

def _atoms(answer):
    nums, names = set(), set()
    for m in _NUM_RE.findall(answer):
        t = m.strip()
        if not t: continue
        digits = re.sub(r'[^\d]', '', t)
        has_unit = bool(re.search(r'%|billion|million|thousand|[BMK]', t, re.I))
        if re.fullmatch(r'(19|20)\d{2}', digits) and not has_unit:
            continue                       # ignore bare publication years
        if has_unit or len(digits) >= 3:   # only 'risky' numerics
            nums.add(t.lower())
    for m in _NAME_RE.findall(answer):
        names.add(m.lower())
    return nums, names

def grounding_report(answer, docs):
    shown = []
    for d in docs:
        shown.append(d.page_content)
        shown.append(d.metadata.get("source", ""))
        shown.append(d.metadata.get("title", ""))
        shown.append(d.metadata.get("arxiv_id", ""))
        _auth = d.metadata.get("authors") or ""
        shown.append(_auth if isinstance(_auth, str) else " ".join(_auth))
    ctx = _gnorm(" ".join(shown))
    ctx_ns = re.sub(r'\s+', '', ctx)
    nums, names = _atoms(answer)
    bad = []
    for n in nums:
        if re.sub(r'\s+', '', n) not in ctx_ns:
            bad.append(("number", n))
    for nm in names:
        surname = nm.replace('et al.', '').replace('et al', '').split()[-1]
        if surname and surname not in ctx:
            bad.append(("name", nm))
    return bad

def deterministic_label_answer(question, docs):
    """Safety net: if the question names 'Table N'/'Figure N' AND we retrieved that
    exact labelled chunk, we can answer from the chunk directly -- no LLM judgment
    involved, so an over-cautious model refusal can't hide a correct retrieval."""
    label = parse_explicit_label(question)
    if not label:
        return None
    field = "table_label" if label.startswith("Table") else "figure_label"
    for d in docs:
        if d.metadata.get(field) == label:
            src = d.metadata.get("source", "")
            # Match the cap format_docs() uses when handing tables to the LLM.
            # 900 chars truncated wide tables (T5 Table 2) mid-grid even though
            # the chunk is already capped at ingest by TABLE_GRID_MAX_CHARS.
            body = trim_text(d.page_content,
                             max_chars=TABLE_GRID_MAX_CHARS + 400,
                             preserve_lines=True)
            return f"From {src}:\n{body}"
    return None
