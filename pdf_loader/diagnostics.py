"""Diagnostics for PDF ingestion artifacts.

Given a diagnostic directory produced by pdf_loader.pdf_loader, summarize per-page
extraction coverage and write optional missing-line reports.

Usage:
    python -m pdf_loader.diagnostics --diag-dir /abs/path/to/diag_run [--report-dir /abs/path/to/diag_run/report]

Outputs:
    - Prints a Markdown table with per-page stats to stdout
    - Writes missing-line reports under report-dir (if provided)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import re


def _norm_text(s: str) -> str:
    s2 = re.sub(r"\s+", " ", (s or "").strip())
    return s2


def _norm_line(s: str) -> str:
    s2 = re.sub(r"\s+", " ", (s or "").strip())
    return s2


def _read_file(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def _collect_pages(diag_dir: Path) -> List[int]:
    merged_dir = diag_dir / "merged"
    if not merged_dir.exists():
        return []
    pages: List[int] = []
    for f in merged_dir.glob("page_*_merged.txt"):
        # filename format: page_XXXX_merged.txt
        try:
            num = int(f.stem.split("_")[1])
            pages.append(num)
        except Exception:
            continue
    return sorted(pages)


def _split_nontrivial_lines(text: str) -> List[str]:
    out: List[str] = []
    for ln in (text or "").splitlines():
        n = _norm_line(ln)
        # only consider lines with some alphanumeric content
        if len(re.sub(r"[^A-Za-z0-9]", "", n)) >= 2:
            out.append(n)
    return out


def analyze_page(diag_dir: Path, page: int, report_dir: Path | None) -> Dict[str, object]:
    t_path = diag_dir / "text" / f"page_{page:04d}_text.txt"
    o_path = diag_dir / "ocr" / f"page_{page:04d}_ocr.txt"
    m_path = diag_dir / "merged" / f"page_{page:04d}_merged.txt"

    text_raw = _read_file(t_path)
    ocr_raw = _read_file(o_path)
    merged_raw = _read_file(m_path)

    merged_norm = _norm_text(merged_raw)
    text_lines = _split_nontrivial_lines(text_raw)
    ocr_lines = _split_nontrivial_lines(ocr_raw)

    def coverage(lines: List[str]) -> Tuple[int, int]:
        if not lines:
            return (0, 0)
        missing = 0
        for ln in lines:
            if ln and (ln not in merged_norm):
                missing += 1
        return (len(lines) - missing, missing)

    ok_t, miss_t = coverage(text_lines)
    ok_o, miss_o = coverage(ocr_lines)

    detected_tables = len(re.findall(r"\[Detected Table ", text_raw))
    used_ocr = "[OCR Merge]" in merged_raw

    if report_dir is not None:
        report_dir.mkdir(parents=True, exist_ok=True)
        if miss_t:
            miss_file = report_dir / f"page_{page:04d}_missing_text_lines.txt"
            # dump only missing lines
            with miss_file.open("w", encoding="utf-8") as f:
                for ln in text_lines:
                    if ln and (ln not in merged_norm):
                        f.write(ln + "\n")
        if miss_o:
            miss_file = report_dir / f"page_{page:04d}_missing_ocr_lines.txt"
            with miss_file.open("w", encoding="utf-8") as f:
                for ln in ocr_lines:
                    if ln and (ln not in merged_norm):
                        f.write(ln + "\n")

    return {
        "page": page,
        "chars_text": len(text_raw or ""),
        "chars_ocr": len(ocr_raw or ""),
        "chars_merged": len(merged_raw or ""),
        "tables_detected": detected_tables,
        "ocr_used": used_ocr,
        "text_lines": len(text_lines),
        "text_lines_covered": ok_t,
        "text_lines_missing": miss_t,
        "ocr_lines": len(ocr_lines),
        "ocr_lines_covered": ok_o,
        "ocr_lines_missing": miss_o,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize PDF ingestion diagnostics")
    ap.add_argument("--diag-dir", required=True, type=Path, help="Diagnostic directory produced by pdf_loader.pdf_loader")
    ap.add_argument("--report-dir", type=Path, default=None, help="Optional directory to write missing-line reports")
    args = ap.parse_args()

    diag_dir: Path = args.diag_dir
    report_dir: Path | None = args.report_dir

    pages = _collect_pages(diag_dir)
    if not pages:
        print("No pages found in diagnostics. Ensure you provided --diagnostic-dir when generating artifacts.")
        return

    rows: List[Dict[str, object]] = []
    for p in pages:
        rows.append(analyze_page(diag_dir, p, report_dir))

    # Print a compact Markdown table
    print("| page | chars_text | chars_ocr | chars_merged | tables | ocr_used | text_cov (ok/miss) | ocr_cov (ok/miss) |")
    print("|---:|---:|---:|---:|---:|:---:|:---:|:---:|")
    for r in rows:
        print(
            f"| {r['page']} | {r['chars_text']} | {r['chars_ocr']} | {r['chars_merged']} | {r['tables_detected']} | {'yes' if r['ocr_used'] else 'no'} | "
            f"{r['text_lines_covered']}/{r['text_lines_missing']} | {r['ocr_lines_covered']}/{r['ocr_lines_missing']} |"
        )


if __name__ == "__main__":
    main()


