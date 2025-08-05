"""Utilities for merging OCR and text, and chunking."""

from typing import List


def merge_texts(page_text: str, ocr_text: str) -> str:
    """Combine visible text with OCR text."""
    raise NotImplementedError("merge_texts is not yet implemented")


def chunk_text(text: str, chunk_size: int) -> List[str]:
    """Split text into chunks of roughly ``chunk_size`` tokens."""
    raise NotImplementedError("chunk_text is not yet implemented")
