"""The request pipeline: guards -> history rewrite -> retrieval -> grounded answer.

Chains are built lazily so importing this module doesn't instantiate a Groq
client (which would make `pytest --collect-only` and `--help` hit the network).
"""

import re
import time

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.output_parsers import StrOutputParser

from app.config import RETRIEVER_K
from app.rag.grounding import deterministic_label_answer, grounding_report
from app.rag.guards import (
    PASTED_REJECT_MSG,
    is_pasted_external_text,
    is_prompt_extraction,
)
from app.rag.prompts import (
    FALLBACK_MSG,
    contextualize_prompt,
    qa_prompt,
    verify_prompt,
)
from app.rag.retrieval import format_docs, select_context_docs, trim_text
from app.rag.store import get_llm, get_llm_regen

class UsageTracker(BaseCallbackHandler):
    """Counts LLM calls + prompt/completion tokens across a single request.
    One instance is created per run_rag() call and passed to every chain
    .invoke via config={"callbacks": [tracker]}."""
    def __init__(self):
        self.calls = self.input_tokens = self.output_tokens = 0
    def on_llm_end(self, response, **kwargs):
        self.calls += 1
        u = (getattr(response, "llm_output", None) or {}).get("token_usage") or {}
        self.input_tokens  += u.get("prompt_tokens", 0)
        self.output_tokens += u.get("completion_tokens", 0)
    @property
    def total_tokens(self):
        return self.input_tokens + self.output_tokens


# --- Chains, built on first use ---------------------------------------------
_chains = {}


def _chain(name):
    if name not in _chains:
        if name == "answer":
            _chains[name] = qa_prompt | get_llm() | StrOutputParser()
        elif name == "answer_regen":
            _chains[name] = qa_prompt | get_llm_regen() | StrOutputParser()
        elif name == "rewrite":
            _chains[name] = contextualize_prompt | get_llm() | StrOutputParser()
        elif name == "verify":
            _chains[name] = verify_prompt | get_llm() | StrOutputParser()
        else:
            raise KeyError(name)
    return _chains[name]

def answer_with_grounding(question, docs, chat_history, debug=False, cfg=None,
                          regenerate=False):
    chain = _chain("answer_regen") if regenerate else _chain("answer")
    answer = chain.invoke({"input": question, "chat_history": chat_history,
                           "context": format_docs(docs)}, config=cfg).strip()
    if debug:
        print("\n[grounding] initial answer:", trim_text(answer, 300))
    # 8B models sometimes refuse even when the exact labelled chunk was retrieved --
    # in that case answer deterministically from the chunk instead of giving up.
    if answer == FALLBACK_MSG:
        det = deterministic_label_answer(question, docs)
        if det:
            if debug:
                print("[grounding] LLM refused but labelled chunk present -> deterministic answer")
            return det
        return answer
    bad = grounding_report(answer, docs)
    if bad:
        if debug:
            print("[grounding] unsupported atoms:", bad)
        answer = _chain("verify").invoke({
            "input": question, "context": format_docs(docs),
            "bad": ", ".join(f"{k}:{v}" for k, v in bad), "fallback": FALLBACK_MSG,
        }, config=cfg).strip()
        if debug:
            print("[grounding] verify-pass answer:", trim_text(answer, 300))
        still_bad = grounding_report(answer, docs)
        if still_bad:
            if debug:
                print("[grounding] still unsupported after verify:", still_bad)
            det = deterministic_label_answer(question, docs)
            answer = det if det else FALLBACK_MSG
    return answer

def compact_history(chat_history, max_turns=4):
    """Keep only recent turns to reduce tokens."""
    return (chat_history or [])[-max_turns:]


_PRONOUN_REFS = {"it","this","that","they","them","these","those","its",
                 "their","he","she","him","her","one","ones"}
_FOLLOWUP_PHRASES = ("same paper","that paper","this paper","same one","same model",
                     "as above","aforementioned","previous","earlier",
                     "elaborate","expand on","go deeper","more detail","in more detail",
                     "tell me more","what about","how about","and what","and how")
_FOLLOWUP_STARTS = ("and ","also ","plus ","then ","what about","how about",
                    "elaborate","expand","continue")

def needs_rewrite(question, chat_history):
    """Rewrite ONLY on genuine dependency on a prior turn. Length is NOT a signal."""
    if not chat_history:
        return False
    q = question.lower().strip()
    tokens = set(re.findall(r"[a-z']+", q))
    if tokens & _PRONOUN_REFS:
        return True
    if any(p in q for p in _FOLLOWUP_PHRASES):
        return True
    if q.startswith(_FOLLOWUP_STARTS):
        return True
    return False


def run_rag(question, chat_history=None, debug=False, regenerate=False):
    """Full request pipeline. Steps, in order:
       1. cheap guards (prompt-extraction, pasted external text) -- no LLM cost
       2. rewrite the question to stand alone IF it depends on chat history
       3. retrieve + select context chunks
       4. answer with grounding verification
       Every return includes: answer, context docs, usage (tokens/calls),
       latency_ms, and mode ("rag" / "rejected")."""
    t0 = time.perf_counter()

    def _ms():
        return round((time.perf_counter() - t0) * 1000)

    # Guard 1: attempts to extract the system prompt -> refuse, spend nothing.
    if is_prompt_extraction(question):
        return {"answer": "I can't share my internal instructions, but I'm happy to "
                          "answer questions about the indexed papers.",
                "context": [], "usage": {}, "latency_ms": _ms(), "mode": "rejected"}

    # Guard 2: pasted external text/error -> out of scope for a RAG system.
    if is_pasted_external_text(question):
        return {"answer": PASTED_REJECT_MSG,
                "context": [], "usage": {}, "latency_ms": _ms(), "mode": "rejected"}

    tracker = UsageTracker()
    cfg = {"callbacks": [tracker]}

    # Robustness: some frontends append the CURRENT question to chat_history before
    # POSTing. Drop a trailing turn that duplicates the question so the rewrite step
    # doesn't see it twice. (If chat_history arrives EMPTY, follow-ups can't resolve
    # at all -- that is a FRONTEND bug, see the notes I gave you.)
    if chat_history:
        last = chat_history[-1]
        last_content = getattr(last, "content", None)
        if last_content is None and isinstance(last, dict):
            last_content = last.get("content")
        if last_content == question:
            chat_history = chat_history[:-1]
    chat_history = compact_history(chat_history)

    is_followup = needs_rewrite(question, chat_history)
    if is_followup:
        standalone_question = _chain("rewrite").invoke({
            "input": question,
            "chat_history": chat_history,
        }, config=cfg).strip()
    else:
        standalone_question = question.strip()
    hist_for_answer = chat_history if is_followup else []

    docs, scored = select_context_docs(standalone_question, final_k=RETRIEVER_K)

    if debug:
        print("=" * 60)
        print("Question:", standalone_question)
        print(f"Candidates: {len(scored)} -> Selected: {len(docs)}")
        print("Papers in selection:", sorted({d.metadata.get('arxiv_id') for d in docs}))
        for i, doc in enumerate(docs, 1):
            print("\nDocument", i, f"[{doc.metadata.get('content_type', 'text')}]")
            print(doc.metadata.get("source"))
            print(trim_text(doc.page_content, 250))

    if not docs:
        answer = FALLBACK_MSG
    else:
        answer = answer_with_grounding(standalone_question, docs, hist_for_answer,
                                       debug=debug, cfg=cfg, regenerate=regenerate)

    if debug:
        print("\nLLM ANSWER:")
        print(answer)

    if answer == FALLBACK_MSG:
        docs = []

    return {"answer": answer, "context": docs, "usage": {
        "llm_calls": tracker.calls,
        "input_tokens": tracker.input_tokens,
        "output_tokens": tracker.output_tokens,
        "total_tokens": tracker.total_tokens,
    }, "latency_ms": _ms(), "mode": "rag"}

