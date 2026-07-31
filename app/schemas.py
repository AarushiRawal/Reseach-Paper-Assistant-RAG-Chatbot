"""Request schemas. Validation happens here, before any handler code runs, so
malformed input is rejected with a 422 and never reaches Chroma or the filesystem.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    # Field(...) = required + validated by FastAPI BEFORE our code runs:
    # empty/whitespace-only or absurdly long questions are rejected with a 422.
    question: str = Field(..., min_length=1, max_length=4000)
    chat_history: list = []   # [{"role": "user"/"assistant", "content": "..."}]
    regenerate: bool = False  # true when the user disliked the answer and wants a retry


class IngestRequest(BaseModel):
    arxiv_id: str = Field(..., min_length=9, max_length=10)


class FeedbackRequest(BaseModel):
    # Explicit schema instead of a raw dict: only these fields, verdict must be
    # exactly "up" or "down", note is optional and capped.
    question: str = Field(..., max_length=4000)
    answer: str = Field(..., max_length=8000)
    verdict: Literal["up", "down"]
    note: Optional[str] = Field(default=None, max_length=1000)
