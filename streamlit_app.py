"""Streamlit frontend.

Talks to the FastAPI backend over HTTP only -- it holds no model, no vector store,
and no API key, so it can run anywhere the backend URL is reachable.

Two things worth noting:
  * chat_history is sent WITHOUT the current question. The backend defends against
    a duplicate trailing turn, but sending it clean keeps the rewrite step honest.
  * figure sources are rendered as actual images via the /figure endpoint rather
    than only showing the generated description.
"""

import os

import requests
import streamlit as st

API_BASE = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000")
TIMEOUT = 180

st.set_page_config(page_title="Research Paper RAG", page_icon="📄", layout="wide")


def api_get(endpoint, **params):
    """`endpoint` (not `path`) because /figure takes a query param literally
    named `path` -- naming both the same collides on the keyword."""
    return requests.get(f"{API_BASE}{endpoint}", params=params, timeout=TIMEOUT)


def api_post(endpoint, payload):
    return requests.post(f"{API_BASE}{endpoint}", json=payload, timeout=TIMEOUT)


def render_sources(sources):
    """Show citations; for figure chunks, display the actual crop."""
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for s in sources:
            st.markdown(f"**{s.get('source') or s.get('title')}**  \n"
                        f"`{s.get('content_type')}` · page {s.get('page')}")
            if s.get("content_type") == "figure" and s.get("image_path"):
                try:
                    resp = api_get("/figure", path=s["image_path"])
                    if resp.status_code == 200:
                        st.image(resp.content, width=520)
                    else:
                        st.caption("(figure image unavailable)")
                except requests.RequestException:
                    st.caption("(figure image unavailable)")
            st.divider()


def render_metrics(payload):
    latency = payload.get("latency_ms")
    usage = payload.get("usage") or {}
    mode = payload.get("mode", "rag")
    cols = st.columns(4)
    cols[0].metric("Latency", f"{latency} ms" if latency is not None else "-")
    cols[1].metric("LLM calls", usage.get("llm_calls", "-"))
    cols[2].metric("Tokens", usage.get("total_tokens", "-"))
    cols[3].metric("Mode", mode)


def send_feedback(question, answer, verdict, note=None):
    try:
        api_post("/feedback", {"question": question, "answer": answer,
                               "verdict": verdict, "note": note})
        st.toast("Feedback recorded.")
    except requests.RequestException:
        st.toast("Could not record feedback.")


# --- sidebar: corpus management ---------------------------------------------

with st.sidebar:
    st.header("Indexed papers")
    try:
        health = api_get("/health").json()
        st.caption(f"Backend OK · {health.get('chunks', '?')} chunks")
    except requests.RequestException:
        st.error(f"Backend unreachable at {API_BASE}")
        st.stop()

    try:
        papers = api_get("/papers").json()
    except requests.RequestException:
        papers = []

    if isinstance(papers, list):
        for p in papers:
            col_a, col_b = st.columns([5, 1])
            col_a.markdown(f"**{p['title'][:60]}**  \n`{p['arxiv_id']}`")
            if col_b.button("✕", key=f"del_{p['arxiv_id']}", help="Delete this paper"):
                requests.delete(f"{API_BASE}/papers/{p['arxiv_id']}", timeout=TIMEOUT)
                st.rerun()

    st.divider()
    new_id = st.text_input("Add a paper (arXiv ID)", placeholder="1706.03762")
    if st.button("Ingest", disabled=not new_id):
        with st.spinner("Ingesting - this downloads, chunks, and embeds the paper..."):
            resp = api_post("/ingest", {"arxiv_id": new_id.strip()})
        if resp.status_code == 200:
            st.success(resp.json()["message"])
            st.rerun()
        else:
            st.error(resp.json().get("detail", "Ingestion failed"))


# --- main chat ---------------------------------------------------------------

st.title("Research Paper RAG Chatbot")
st.caption("Answers are grounded in the indexed arXiv papers only.")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_payload" not in st.session_state:
    st.session_state.last_payload = None

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


def ask(question, regenerate=False):
    # History EXCLUDES the current question.
    history = [{"role": m["role"], "content": m["content"]}
               for m in st.session_state.messages]
    if regenerate and history and history[-1]["role"] == "assistant":
        history = history[:-1]

    with st.spinner("Retrieving and answering..."):
        resp = api_post("/chat", {"question": question, "chat_history": history,
                                  "regenerate": regenerate})

    if resp.status_code != 200:
        detail = resp.json().get("detail", "Request failed")
        st.error(detail)
        return None
    return resp.json()


prompt = st.chat_input("Ask about the indexed papers...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    payload = ask(prompt)
    if payload:
        st.session_state.messages.append({"role": "assistant", "content": payload["answer"]})
        st.session_state.last_payload = {"question": prompt, **payload}
        with st.chat_message("assistant"):
            st.markdown(payload["answer"])
            render_metrics(payload)
            render_sources(payload.get("sources"))

# --- feedback + regenerate on the most recent answer -------------------------

if st.session_state.last_payload:
    lp = st.session_state.last_payload
    c1, c2, c3 = st.columns([1, 1, 6])
    if c1.button("👍"):
        send_feedback(lp["question"], lp["answer"], "up")
    if c2.button("👎"):
        send_feedback(lp["question"], lp["answer"], "down")
    if c3.button("Regenerate answer"):
        payload = ask(lp["question"], regenerate=True)
        if payload:
            if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
                st.session_state.messages[-1]["content"] = payload["answer"]
            st.session_state.last_payload = {"question": lp["question"], **payload}
            st.rerun()
