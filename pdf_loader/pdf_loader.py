"""Orchestrator for PDF ingestion and OCR.

This module coordinates extraction of visible text, image extraction,
OCR processing, merging, and chunking to produce JSON output ready for
embedding. Supports page ranges and diagnostic artifact output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import sys
import time

from .pdf_text import extract_text
from .pdf_images import extract_images
from .ocr import ocr_image_paths
from .merge import merge_texts, chunk_text

def _parse_page_arg(page_range: Optional[str]) -> Optional[List[int]]:
    if not page_range:
        return None
    pages: List[int] = []
    for part in str(page_range).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            try:
                start = int(a)
                end = int(b)
                if start <= end:
                    pages.extend(list(range(start, end + 1)))
                else:
                    pages.extend(list(range(end, start + 1)))
            except Exception:
                continue
        else:
            try:
                pages.append(int(part))
            except Exception:
                continue
    # dedupe and sort
    return sorted({p for p in pages if p > 0}) or None


def process_pdf(
    pdf_path: Path,
    output_json: Path,
    image_dir: Path,
    *,
    chunk_size: int = 400,
    page_range: Optional[str] = None,
    render_pages_dpi: Optional[int] = 300,
    ocr_language: str = "eng",
    ocr_psm: int = 6,
    ocr_oem: int = 3,
    diagnostic_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Process a PDF into text chunks and write them to a JSON file.

    Parameters
    ----------
    pdf_path: Path
        Path to the PDF document.
    output_json: Path
        Destination JSON file for serialized chunks.
    image_dir: Path
        Directory where extracted images will be saved.
    chunk_size: int
        Approximate number of tokens per chunk.
    page_range: Optional[str]
        Page spec like "10-20,35,40-42" (1-based). When None process all.
    render_pages_dpi: Optional[int]
        If set, render each page to a bitmap at this DPI to improve OCR capture of vector tables.
    ocr_language, ocr_psm, ocr_oem: OCR engine configuration.
    diagnostic_dir: Optional[Path]
        If provided, save per-page artifacts (render, text, ocr, merged).
    """
    pages = _parse_page_arg(page_range)

    # Ensure diagnostic dir exists if requested
    if diagnostic_dir is not None:
        diagnostic_dir.mkdir(parents=True, exist_ok=True)

    print(f"[pdf-loader] Extracting text from {len(pages) if pages else 'all'} pages...", flush=True)
    page_texts = extract_text(
        pdf_path,
        page_numbers=pages,
        include_tables=True,
        diagnostic_dir=(diagnostic_dir / "text") if diagnostic_dir else None,
    )
    print(f"[pdf-loader] Text extraction complete. Extracting images...", flush=True)
    page_images = extract_images(
        pdf_path,
        image_dir,
        page_numbers=pages,
        render_pages_dpi=render_pages_dpi,
        diagnostic_dir=(diagnostic_dir / "renders") if diagnostic_dir else None,
    )
    print(f"[pdf-loader] Image extraction complete. Processing pages...", flush=True)

    chunks: List[Dict[str, Any]] = []
    chunk_id = 0

    # Iterate over the union of pages that have text and/or images (renders)
    all_pages = sorted(set(page_texts.keys()) | set(page_images.keys()))
    total_pages = len(all_pages)
    t_start = time.time()
    for idx, page_num in enumerate(all_pages, start=1):
        if total_pages:
            elapsed = time.time() - t_start
            avg = elapsed / max(1, (idx - 1)) if idx > 1 else 0.0
            remaining = (total_pages - idx + 1) * avg if avg > 0 else 0.0
            eta_min = int(remaining // 60)
            eta_sec = int(remaining % 60)
            print(f"[pdf] {idx}/{total_pages} (p={page_num}) ETA ~{eta_min}m {eta_sec}s", flush=True)
        text = page_texts.get(page_num, "")
        images = page_images.get(page_num, [])
        # Run OCR on all images including full-page renders
        ocr_text = ocr_image_paths(images, language=ocr_language, psm=ocr_psm, oem=ocr_oem)

        # Save diagnostics before/after merge
        if diagnostic_dir is not None:
            (diagnostic_dir / "ocr").mkdir(parents=True, exist_ok=True)
            (diagnostic_dir / "merged").mkdir(parents=True, exist_ok=True)
            try:
                (diagnostic_dir / "ocr" / f"page_{page_num:04d}_ocr.txt").write_text(ocr_text or "", encoding="utf-8")
            except Exception:
                pass

        merged = merge_texts(text, ocr_text)

        if diagnostic_dir is not None:
            try:
                (diagnostic_dir / "merged" / f"page_{page_num:04d}_merged.txt").write_text(merged or "", encoding="utf-8")
            except Exception:
                pass

        for chunk in chunk_text(merged, chunk_size):
            tokens = len(chunk.split())
            chunks.append({
                "chunk_id": chunk_id,
                "page": page_num,
                "text": chunk,
                "tokens": tokens,
                "source": str(pdf_path)
            })
            chunk_id += 1

    with output_json.open("w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    return chunks


def build_arg_parser() -> argparse.ArgumentParser:
    """Create an argument parser for the CLI."""
    parser = argparse.ArgumentParser(description="Process a PDF into JSON chunks.")
    parser.add_argument("pdf_path", type=Path, help="Path to the PDF file.")
    parser.add_argument("--output", type=Path, default=Path("pdf_chunks.json"), help="Path for the output JSON file.")
    parser.add_argument("--image-dir", type=Path, default=Path("data/images"), help="Directory to store extracted images.")
    parser.add_argument("--chunk-size", type=int, default=400, help="Approximate number of tokens per chunk.")
    parser.add_argument("--pages", type=str, default=None, help="Page selection like '10-20,35,40-42'. 1-based.")
    parser.add_argument("--render-dpi", type=int, default=300, help="DPI for full-page rendering used in OCR (0 to disable).")
    parser.add_argument("--ocr-lang", type=str, default="eng", help="Tesseract languages, e.g., 'eng' or 'eng+osd'.")
    parser.add_argument("--ocr-psm", type=int, default=6, help="Tesseract PSM mode.")
    parser.add_argument("--ocr-oem", type=int, default=3, help="Tesseract OEM mode.")
    parser.add_argument("--diagnostic-dir", type=Path, default=None, help="Directory to store per-page diagnostics.")
    return parser


def main() -> None:
    """CLI entry point."""
    parser = build_arg_parser()
    args = parser.parse_args()
    process_pdf(
        args.pdf_path,
        args.output,
        args.image_dir,
        chunk_size=args.chunk_size,
        page_range=args.pages,
        render_pages_dpi=(args.render_dpi if args.render_dpi and args.render_dpi > 0 else None),
        ocr_language=args.ocr_lang,
        ocr_psm=args.ocr_psm,
        ocr_oem=args.ocr_oem,
        diagnostic_dir=args.diagnostic_dir,
    )


if __name__ == "__main__":
    main()
