"""Figure extraction: locate figure regions by caption, render, OCR + caption them.

Each figure region becomes (a) a searchable TEXT chunk containing the PDF caption,
any OCR'd in-image text, and the MoonDream visual description, and (b) a PNG crop
saved under PDF_DIR so the /figure endpoint can serve the actual image to the UI.
"""

import hashlib
import os
import re

import fitz
from langchain_core.documents import Document

from app.config import IMAGE_RAG_POLICY, PDF_DIR
from app.ingestion.cache import load_cache, save_cache
from app.ingestion.vision import describe_image_local, ocr_image_local

def find_figure_regions(page, margin_above=None, margin_below=None):
    """Find figure regions by locating Figure/Fig captions and cropping above them."""
    margin_above = margin_above or IMAGE_RAG_POLICY["caption_margin_above"]
    margin_below = margin_below or IMAGE_RAG_POLICY["caption_margin_below"]
    regions = []
    caption_pattern = re.compile(r"^(Figure|Fig\.)\s*\d+", re.IGNORECASE)

    for block in page.get_text("dict").get("blocks", []):
        if "lines" not in block:
            continue
        block_text = "".join(
            span["text"] for line in block["lines"] for span in line["spans"]
        ).strip()
        if caption_pattern.match(block_text):
            caption_bbox = fitz.Rect(block["bbox"])
            fig_rect = fitz.Rect(
                0,
                max(0, caption_bbox.y0 - margin_above),
                page.rect.width,
                min(page.rect.height, caption_bbox.y0 + margin_below),
            )
            regions.append({
                "rect": fig_rect,
                "caption_text": block_text,
            })
    return regions

def build_figure_rag_text(title, arxiv_id, page_num, caption_text, ocr_text, vision_caption):
    parts = [
        f"Figure/Image from research paper: {title}",
        f"arXiv ID: {arxiv_id}",
        f"Page: {page_num}",
    ]
    if caption_text:
        parts.append(f"PDF figure caption: {caption_text}")
    if ocr_text:
        parts.append(f"Text detected inside image/chart/diagram: {ocr_text}")
    if vision_caption:
        parts.append(f"Visual interpretation: {vision_caption}")
    return "\n".join(parts)

def extract_and_describe_figures(filepath, paper):
    """Convert selected chart/diagram/image regions into searchable RAG text chunks.

    This is global-policy driven: every new paper uses the same policy. OCR is cheap;
    vision captioning is cached and bounded by max_images_per_paper.
    """
    if not IMAGE_RAG_POLICY.get("enabled", True):
        return []

    doc = fitz.open(filepath)
    figure_docs = []
    cache = load_cache()
    seen_hashes = set()
    n_processed = 0

    max_images = IMAGE_RAG_POLICY["max_images_per_paper"]
    min_width = IMAGE_RAG_POLICY["min_width"]
    min_height = IMAGE_RAG_POLICY["min_height"]
    render_dpi = IMAGE_RAG_POLICY["render_dpi"]
    use_ocr = IMAGE_RAG_POLICY["use_ocr"]
    use_vision = IMAGE_RAG_POLICY["use_vision_caption"]

    for page_num, page in enumerate(doc, start=1):
        if n_processed >= max_images:
            break

        regions = find_figure_regions(page)
        for region_index, region_info in enumerate(regions):
            if n_processed >= max_images:
                break

            rect = region_info["rect"]
            caption_text = region_info.get("caption_text", "")
            if rect.height < 50 or rect.width < 50:
                continue

            pix = page.get_pixmap(clip=rect, dpi=render_dpi)
            if pix.width < min_width or pix.height < min_height:
                continue

            image_bytes = pix.tobytes("png")
            img_hash = hashlib.md5(image_bytes).hexdigest()
            if img_hash in seen_hashes:
                continue
            seen_hashes.add(img_hash)

            arxiv_id = paper["arxiv_id"]
            title = paper["title"]
            img_path = os.path.join(PDF_DIR, f"{arxiv_id}_p{page_num}_fig{region_index}.png")
            with open(img_path, "wb") as f:
                f.write(image_bytes)

            cache_key = f"{arxiv_id}_p{page_num}_fig{region_index}_{img_hash}"
            cached = cache.get(cache_key, {})

            if use_ocr and "ocr_text" in cached:
                ocr_text = cached["ocr_text"]
            elif use_ocr:
                ocr_text = ocr_image_local(image_bytes)
            else:
                ocr_text = ""

            if use_vision and "vision_caption" in cached:
                vision_caption = cached["vision_caption"]
                print(f"    (cached image RAG) p{page_num} fig{region_index}")
            elif use_vision:
                try:
                    vision_caption = describe_image_local(image_bytes)
                    print(f"    (new image RAG) p{page_num} fig{region_index}")
                except Exception as e:
                    print(f"    Vision caption failed p{page_num} fig{region_index}: {e}")
                    vision_caption = ""
            else:
                vision_caption = ""

            cache[cache_key] = {
                "ocr_text": ocr_text,
                "vision_caption": vision_caption,
                "caption_text": caption_text,
                "image_path": img_path,
            }
            save_cache(cache)

            page_content = build_figure_rag_text(
                title=title,
                arxiv_id=arxiv_id,
                page_num=page_num,
                caption_text=caption_text,
                ocr_text=ocr_text,
                vision_caption=vision_caption,
            )

            # Only add useful image chunks. Caption-only placeholders are allowed,
            # but empty OCR + empty vision + empty caption is noise.
            if not (caption_text or ocr_text or vision_caption):
                continue

            figure_docs.append(Document(
                page_content=page_content,
                metadata={
                    "arxiv_id": arxiv_id,
                    "title": title,
                    "authors": paper["authors"],
                    "page": page_num,
                    "figure_label": (
                        (lambda m: f"Figure {m.group(1)}" if m else "")(
                            re.match(r"(?:Figure|Fig\.?)\s*(\d+)", caption_text or "", re.IGNORECASE))
                    ),
                    "source": f"{title[:50]} (arXiv:{arxiv_id}, p.{page_num}, figure/image)",
                    "content_type": "figure",
                    "image_path": img_path,
                    "has_ocr_text": bool(ocr_text),
                    "has_vision_caption": bool(vision_caption),
                },
            ))
            n_processed += 1

    return figure_docs
