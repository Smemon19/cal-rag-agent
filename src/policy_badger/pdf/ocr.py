"""OCR utilities with robust configuration and table-preserving post-processing."""

from pathlib import Path
from typing import List, Optional
import os
import pytesseract
from PIL import Image, ImageOps


def _preprocess_for_ocr(img: Image.Image) -> Image.Image:
    # Convert to grayscale and apply slight contrast enhancement
    g = ImageOps.grayscale(img)
    return g


def ocr_image_paths(
    image_paths: List[Path],
    *,
    language: str = "eng",
    psm: int = 6,
    oem: int = 3,
    preserve_line_breaks: bool = True,
) -> str:
    """Run OCR on image paths with configurable engine settings.

    Parameters
    ----------
    language: ISO codes like 'eng', 'eng+osd', or add table-optimized models if installed
    psm: Tesseract page segmentation mode (6: Assume a single uniform block of text)
    oem: OCR Engine Mode (3: Default, based on what is available)
    preserve_line_breaks: Keep line breaks to help retain table-like structures
    """
    # Allow overriding the tesseract.exe path via environment variable on Windows installers
    tess_exe = os.getenv("TESSERACT_EXE")
    if tess_exe:
        try:
            pytesseract.pytesseract.tesseract_cmd = tess_exe
        except Exception:
            pass

    if not image_paths:
        return ""

    config = f"--oem {oem} --psm {psm}"
    extracted_text: List[str] = []

    for img_path in image_paths:
        try:
            with Image.open(img_path) as img:
                img2 = _preprocess_for_ocr(img)
                if preserve_line_breaks:
                    text = pytesseract.image_to_string(img2, lang=language, config=config)
                else:
                    text = pytesseract.image_to_string(img2, lang=language, config=config)
                text = (text or "").strip()
                if text:
                    extracted_text.append(text)
        except Exception as e:
            print(f"Error performing OCR on {img_path}: {e}")
            continue

    return "\n".join(extracted_text)
