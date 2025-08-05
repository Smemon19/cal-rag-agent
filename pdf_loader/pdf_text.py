"""Utilities for extracting visible text from PDF pages."""

from pathlib import Path
from typing import Dict


def extract_text(pdf_path: Path) -> Dict[int, str]:
    """Return mapping of page numbers to visible text.

    Parameters
    ----------
    pdf_path: Path
        Path to the PDF file.
    """
    raise NotImplementedError("extract_text is not yet implemented")
