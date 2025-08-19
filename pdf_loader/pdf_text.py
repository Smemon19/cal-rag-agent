"""Utilities for extracting visible text and tables from PDF pages."""

from pathlib import Path
from typing import Dict, List, Optional
import fitz  # PyMuPDF


def _parse_page_selection(total_pages: int, page_numbers: Optional[List[int]] = None) -> List[int]:
    if page_numbers:
        # sanitize and clamp to valid 1-based page numbers
        pages = sorted({p for p in page_numbers if isinstance(p, int) and 1 <= p <= total_pages})
        return pages
    return list(range(1, total_pages + 1))


def _extract_text_blocks(page: fitz.Page) -> str:
    # Use block extraction to better preserve columns and spacing
    blocks = page.get_text("blocks", sort=True)
    lines: List[str] = []
    for b in blocks:
        # b: (x0, y0, x1, y1, "text", block_no, block_type, ...)
        if len(b) >= 5:
            txt = (b[4] or "").rstrip()
            if txt:
                lines.append(txt)
    return "\n\n".join(lines).strip()


def _extract_tables_markdown(page: fitz.Page) -> str:
    # Try to detect vector-drawn tables and export as simple Markdown
    try:
        tf = page.find_tables()
        tables = getattr(tf, "tables", []) if tf else []
    except Exception:
        tables = []

    md_parts: List[str] = []
    for t_index, t in enumerate(tables, start=1):
        try:
            matrix = t.extract()
        except Exception:
            matrix = None
        if not matrix:
            continue
        # Build a minimal Markdown table; assume first row as header when plausible
        rows = [[(cell or "").strip() for cell in row] for row in matrix]
        if not rows:
            continue
        header = rows[0]
        md_parts.append(f"[Detected Table {t_index}]\n")
        md_parts.append("| " + " | ".join(header) + " |")
        md_parts.append("| " + " | ".join(["---" for _ in header]) + " |")
        for r in rows[1:]:
            md_parts.append("| " + " | ".join(r) + " |")
        md_parts.append("")
    return "\n".join(md_parts).strip()


def extract_text(
    pdf_path: Path,
    *,
    page_numbers: Optional[List[int]] = None,
    include_tables: bool = True,
    diagnostic_dir: Optional[Path] = None,
) -> Dict[int, str]:
    """Return mapping of 1-based page numbers to text, optionally including table markdown.

    Parameters
    ----------
    pdf_path: Path
        Path to the PDF file.
    page_numbers: Optional[List[int]]
        Specific 1-based page numbers to process. When None, process all pages.
    include_tables: bool
        When True, append detected tables as Markdown to each page's text.
    diagnostic_dir: Optional[Path]
        When provided, write a per-page raw text dump for diagnostics: page_XXX_text.txt
    """
    doc = fitz.open(pdf_path)
    try:
        pages = _parse_page_selection(len(doc), page_numbers)
        page_texts: Dict[int, str] = {}

        if diagnostic_dir is not None:
            diagnostic_dir.mkdir(parents=True, exist_ok=True)

        total = len(pages)
        for idx, pnum in enumerate(pages, start=1):
            try:
                print(f"[pdf-text] {idx}/{total} (p={pnum})", flush=True)
            except Exception:
                pass
            page = doc[pnum - 1]
            text_blocks = _extract_text_blocks(page)
            page_text = text_blocks

            if include_tables:
                tables_md = _extract_tables_markdown(page)
                if tables_md:
                    if page_text:
                        page_text = f"{page_text}\n\n{tables_md}"
                    else:
                        page_text = tables_md

            page_text = (page_text or "").strip()
            if page_text:
                page_texts[pnum] = page_text

            if diagnostic_dir is not None:
                out = diagnostic_dir / f"page_{pnum:04d}_text.txt"
                try:
                    out.write_text(page_text, encoding="utf-8")
                except Exception:
                    pass
        return page_texts
    finally:
        doc.close()
