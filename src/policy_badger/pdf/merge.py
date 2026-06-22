"""Utilities for merging OCR and text, and chunking."""

from typing import List
import re


def merge_texts(page_text: str, ocr_text: str) -> str:
    """Combine visible text with OCR text, deduplicating overlapping lines.

    Strategy:
    - Always include both sources to avoid missing content
    - If OCR lines are already present verbatim in visible text, skip duplicates
    - Preserve line breaks to retain table-like alignment as much as possible
    """
    pt = (page_text or "").strip()
    ot = (ocr_text or "").strip()
    if not pt and not ot:
        return ""
    if not pt:
        return ot
    if not ot:
        return pt

    existing_lines = {ln.strip() for ln in pt.splitlines() if ln.strip()}
    merged_lines: List[str] = []

    for ln in ot.splitlines():
        s = ln.rstrip()
        if not s:
            merged_lines.append("")
            continue
        if s.strip() in existing_lines:
            # skip duplicate
            continue
        merged_lines.append(s)

    if merged_lines:
        return pt + "\n\n[OCR Merge]\n" + "\n".join(merged_lines)
    return pt


def chunk_text(text: str, chunk_size: int) -> List[str]:
    """Split text into chunks of roughly ``chunk_size`` tokens."""
    if not text:
        return []
    
    # Simple token-based splitting (words)
    words = text.split()
    chunks = []
    current_chunk = []
    current_size = 0
    
    for word in words:
        word_size = len(word) + 1  # +1 for space
        
        if current_size + word_size > chunk_size and current_chunk:
            # Save current chunk and start new one
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_size = word_size
        else:
            current_chunk.append(word)
            current_size += word_size
    
    # Add the last chunk if it exists
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks
