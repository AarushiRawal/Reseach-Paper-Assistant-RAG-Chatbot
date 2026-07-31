"""Cheap pre-retrieval guards. Both run before any embedding or LLM call, so a
rejected request costs nothing.

Guard 1 - prompt extraction: refuses attempts to read back the system prompt.
Guard 2 - pasted external text: this is a RAG chatbot scoped to the indexed
papers, so a pasted traceback or an outside paragraph is rejected rather than
answered ungrounded.
"""

import re

# --- Prompt-extraction / injection guard (cheap, runs before any LLM call) ---
_INJECTION_RE = re.compile(
    r"(system\s+prompt|your\s+(instructions|prompt|rules|guidelines)|"
    r"reveal.*(prompt|instructions)|ignore\s+(all\s+)?(previous|prior|above)|"
    r"what\s+(are|were)\s+you\s+(told|instructed)|repeat.*(above|instructions))", re.I)

def is_prompt_extraction(q):
    return bool(_INJECTION_RE.search(q or ""))


# --- Pasted-text / off-corpus input guard ------------------------------------
# DESIGN DECISION (v16): this is a RAG chatbot, scoped to the indexed papers.
# If the user pastes a block of EXTERNAL text (an error traceback, a paragraph
# copied from somewhere) and asks us to explain it, we REJECT with a clear
# message instead of answering ungrounded.
#
# Why detect it BEFORE retrieval instead of letting it hit the normal fallback?
#   1. Zero wasted embedding/LLM calls (rejection costs nothing).
#   2. Pasted text that vaguely resembles an indexed paper would otherwise
#      retrieve chunks and produce a confidently WRONG "explanation".
#
# Detection rule (deliberately simple -- a keyword/shape check, not a model):
#   - it looks like an error dump (traceback markers), OR
#   - it contains an "explain/summarize this"-style instruction AND a pasted
#     BLOCK (long message, several newlines, or a ``` code fence).
# "Explain self-attention" has the word "explain" but no pasted block, so it
# still goes through normal RAG. Known limitation: a determined user can
# phrase around this; acceptable for the demo scope.

PASTED_REJECT_MSG = (
    "I can only answer questions about the papers currently indexed in this "
    "system, so I can't explain external pasted text. You can ask me about the "
    "indexed papers, or ingest the paper this text comes from via the add-paper "
    "option."
)

_ERROR_MARKERS = ("traceback (most recent call last)", 'file "', "error:",
                  "exception:", "errno", "stack trace")
_EXPLAIN_WORDS = ("explain", "what does this mean", "summarize this",
                  "summarise this", "simplify this", "what is this")

def is_pasted_external_text(q):
    """True if the message looks like pasted external content to explain."""
    q_low = (q or "").lower()
    if any(m in q_low for m in _ERROR_MARKERS):
        return True
    has_block = len(q_low) > 500 or q_low.count("\n") >= 3 or "```" in q_low
    has_instruction = any(w in q_low for w in _EXPLAIN_WORDS)
    return has_block and has_instruction
