"""Orchestrator for PDF ingestion and OCR.

This module coordinates extraction of visible text, image extraction,
OCR processing, merging, and chunking to produce JSON output ready for
embedding.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any

from .pdf_text import extract_text
from .pdf_images import extract_images
from .ocr import ocr_image_paths
from .merge import merge_texts, chunk_text


def process_pdf(pdf_path: Path, output_json: Path, image_dir: Path, chunk_size: int = 400) -> List[Dict[str, Any]]:
    """Process a PDF into text chunks and write them to a JSON file.

    Parameters
    ----------
    pdf_path: Path
        Path to the PDF document.
    output_json: Path
        Destination JSON file for serialized chunks.
    image_dir: Path
        Directory where extracted images will be saved.
    chunk_size: int, optional
        Approximate number of tokens per chunk.
    """
    page_texts = extract_text(pdf_path)
    page_images = extract_images(pdf_path, image_dir)

    chunks: List[Dict[str, Any]] = []
    chunk_id = 0

    for page_num, text in page_texts.items():
        images = page_images.get(page_num, [])
        ocr_text = ocr_image_paths(images)
        merged = merge_texts(text, ocr_text)
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
    return parser


def main() -> None:
    """CLI entry point."""
    parser = build_arg_parser()
    args = parser.parse_args()
    process_pdf(args.pdf_path, args.output, args.image_dir, args.chunk_size)


if __name__ == "__main__":
    main()
