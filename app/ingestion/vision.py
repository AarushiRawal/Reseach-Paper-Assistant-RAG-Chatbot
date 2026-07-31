"""Local figure understanding: MoonDream2 vision captioning + Tesseract OCR.

THE ONLY GPU-DEPENDENT MODULE IN THE PROJECT.

MoonDream2 (~1.8B params) runs on CUDA when a GPU is available and falls back to
CPU otherwise -- the fallback works, it is just slower (roughly 10-20s per figure
on CPU vs ~1-2s on a modern GPU). This runs at INGESTION time only, capped at
`max_images_per_paper` figures per paper and cached afterwards, so a full 11-paper
index is a one-off cost of at most ~66 captions. Serving a query never touches it.

If you have no GPU and want ingestion to finish fast, set
IMAGE_RAG_POLICY["use_vision_caption"] = False in config.py -- figure chunks then
carry their PDF caption + OCR text only, which is still searchable.
"""

import io

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import pytesseract
except Exception:  # pragma: no cover - optional system dependency
    pytesseract = None

from app.config import TESSERACT_CMD

if pytesseract is not None and TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
def get_device():
    """cuda when available, else cpu. Exposed so scripts can log what they got."""
    return "cuda" if torch.cuda.is_available() else "cpu"

_moondream_model = None
_moondream_tokenizer = None


def get_moondream():
    global _moondream_model, _moondream_tokenizer
    if _moondream_model is None:
        _moondream_model = AutoModelForCausalLM.from_pretrained(
            "vikhyatk/moondream2", trust_remote_code=True
        ).to(get_device())
        _moondream_tokenizer = AutoTokenizer.from_pretrained("vikhyatk/moondream2")
    return _moondream_model, _moondream_tokenizer


def describe_image_local(image_bytes):
    """Generate a compact figure/chart/diagram caption using MoonDream locally."""
    model, tokenizer = get_moondream()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    enc = model.encode_image(img)
    return model.answer_question(
        enc,
        "This is a figure, chart, diagram, or screenshot from a research paper. "
        "Describe the important visual information in 2-4 concise sentences. "
        "Include chart axes, legends, architecture blocks, arrows, trends, or comparisons if visible.",
        tokenizer,
    )


def ocr_image_local(image_bytes):
    """Extract embedded text from a figure image using local Tesseract OCR."""
    if pytesseract is None:
        return ""
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        text = pytesseract.image_to_string(img, config="--psm 6")
        return " ".join(text.split())
    except Exception as e:
        print(f"    OCR failed: {e}")
        return ""
