"""Utilities for extracting images from PDF pages."""

from pathlib import Path
from typing import Dict, List


def extract_images(pdf_path: Path, image_dir: Path) -> Dict[int, List[Path]]:
    """Extract images from a PDF and return mapping of page number to image paths."""
    raise NotImplementedError("extract_images is not yet implemented")
