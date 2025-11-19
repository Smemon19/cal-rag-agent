import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Add project root to path so we can import utils
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from google.cloud import bigquery

# Reuse existing utilities for metadata and embeddings
from utils import (
    make_chunk_metadata,
    resolve_embedding_backend_and_model,
    create_embedding_function,
)

import re


def ensure_dataset(client: bigquery.Client, project: str, dataset: str, location: str) -> None:
    dataset_id = f"{project}.{dataset}"
    try:
        client.get_dataset(dataset_id)
        return
    except Exception:
        pass
    ds = bigquery.Dataset(dataset_id)
    ds.location = location
    client.create_dataset(ds, exists_ok=True)


def ensure_table(client: bigquery.Client, project: str, dataset: str, table: str) -> None:
    table_id = f"{project}.{dataset}.{table}"
    try:
        client.get_table(table_id)
        return
    except Exception:
        pass
    schema = [
        bigquery.SchemaField("chunk_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("content", "STRING"),
        bigquery.SchemaField("embedding", "FLOAT64", mode="REPEATED"),
        bigquery.SchemaField("source_url", "STRING"),
        bigquery.SchemaField("source_type", "STRING"),
        bigquery.SchemaField("section_path", "STRING"),
        bigquery.SchemaField("headers", "STRING"),
        bigquery.SchemaField("page_number", "INT64"),
        bigquery.SchemaField("title", "STRING"),
        bigquery.SchemaField("mime_type", "STRING"),
        bigquery.SchemaField("char_count", "INT64"),
        bigquery.SchemaField("word_count", "INT64"),
        bigquery.SchemaField("content_preview", "STRING"),
        bigquery.SchemaField("embedding_backend", "STRING"),
        bigquery.SchemaField("embedding_model", "STRING"),
        bigquery.SchemaField("inserted_at", "TIMESTAMP"),
    ]
    table_obj = bigquery.Table(table_id, schema=schema)
    client.create_table(table_obj, exists_ok=True)


def smart_chunk_markdown(markdown: str, max_len: int = 1000, overlap_chars: int = 150) -> List[str]:
    import re

    def split_by_header(md: str, header_pattern: str) -> List[str]:
        indices = [m.start() for m in re.finditer(header_pattern, md, re.MULTILINE)]
        indices.append(len(md))
        return [md[indices[i]:indices[i + 1]].strip()
                for i in range(len(indices) - 1)
                if md[indices[i]:indices[i + 1]].strip()]

    if not markdown:
        return []

    chunks: List[str] = []
    for h1 in split_by_header(markdown, r'^# .+$'):
        if len(h1) > max_len:
            for h2 in split_by_header(h1, r'^## .+$'):
                if len(h2) > max_len:
                    for h3 in split_by_header(h2, r'^### .+$'):
                        if len(h3) > max_len:
                            step = max(1, max_len - max(0, overlap_chars))
                            i = 0
                            while i < len(h3):
                                chunk = h3[i:i + max_len].strip()
                                if chunk:
                                    chunks.append(chunk)
                                i += step
                        else:
                            chunks.append(h3)
                else:
                    chunks.append(h2)
        else:
            chunks.append(h1)
    if not chunks:
        chunks = [markdown]

    final_chunks: List[str] = []
    step = max(1, max_len - max(0, overlap_chars))
    for c in chunks:
        if len(c) > max_len:
            i = 0
            while i < len(c):
                piece = c[i:i + max_len].strip()
                if piece:
                    final_chunks.append(piece)
                i += step
        else:
            final_chunks.append(c)
    return [c for c in final_chunks if c]


def read_pdf_chunks(file_path: Path,
                    page_range: Optional[str] = None,
                    render_dpi: int = 300,
                    ocr_lang: str = "eng",
                    ocr_psm: int = 6,
                    ocr_oem: int = 3) -> Tuple[List[str], List[Optional[int]], str]:
    try:
        from pdf_loader.pdf_loader import process_pdf
    except Exception as e:
        raise RuntimeError(f"PDF dependencies not installed or import failed: {e}")

    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        output_json = temp_path / "pdf_chunks.json"
        image_dir = temp_path / "images"
        chunks = process_pdf(
            file_path,
            output_json,
            image_dir,
            chunk_size=1000,
            page_range=page_range,
            render_pages_dpi=render_dpi,
            ocr_language=ocr_lang,
            ocr_psm=ocr_psm,
            ocr_oem=ocr_oem,
            diagnostic_dir=None,
        )
        # Title from PDF metadata (best-effort)
        title = ""
        try:
            import fitz  # PyMuPDF
            with fitz.open(file_path) as doc:
                title = (doc.metadata or {}).get("title") or ""
        except Exception:
            title = file_path.stem
        texts = [str(ch.get("text") or "") for ch in chunks]
        pages = [int(ch.get("page") or 0) or None for ch in chunks]
        return texts, pages, title or file_path.stem


def enumerate_files(root: Path) -> List[Path]:
    exts = {".pdf", ".txt", ".md", ".markdown"}
    files: List[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    return files


def _looks_like_section_number(line: str) -> Optional[str]:
    m = re.match(r"^\s*(?:Section\s+)?(\d+(?:\.\d+)+)\b", line, flags=re.IGNORECASE)
    if m:
        return m.group(1)
    return None


def _looks_like_table_number(line: str) -> Optional[str]:
    m = re.match(r"^\s*(Table\s+\d+(?:\.\d+)*)\b", line, flags=re.IGNORECASE)
    if m:
        return m.group(1)
    return None


def _looks_like_upper_heading(line: str) -> bool:
    s = line.strip()
    if len(s) < 3 or len(s) > 120:
        return False
    # Count letters and uppercase ratio
    letters = [ch for ch in s if ch.isalpha()]
    if not letters:
        return False
    upper = sum(1 for ch in letters if ch.isupper())
    ratio = upper / max(1, len(letters))
    # Avoid lines that end with periods often (sentences)
    if s.endswith("."):
        return False
    # Consider uppercase-like if ratio high
    return ratio >= 0.8


def extract_headers_for_pdf_chunk(text: str) -> Tuple[str, str]:
    """
    Infer headers and a section_path for a PDF chunk by scanning its first lines.
    Returns (headers_text, section_path). Empty strings if nothing found.
    Heuristics:
      - Prefer numeric section tokens (e.g., 1507.9.6) and table numbers.
      - Fallback to prominent uppercase headings.
    """
    if not text:
        return "", ""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    candidates: List[str] = []

    # Scan the first ~40 lines or total lines if shorter
    scan_upto = min(40, len(lines))
    for i in range(scan_upto):
        ln = lines[i]
        sec = _looks_like_section_number(ln)
        if sec:
            # Include the line as heading: "Section 1507.9.6 ..." or just the number
            candidates.append(ln)
            continue
        tab = _looks_like_table_number(ln)
        if tab:
            candidates.append(ln)
            continue
        if _looks_like_upper_heading(ln):
            candidates.append(ln)

    # Dedup while preserving order and keep at most 3
    seen = set()
    uniq = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
        if len(uniq) >= 3:
            break

    headers_text = "; ".join(uniq)

    # Build section_path using more structured tokens if present
    parts: List[str] = []
    # 1) Primary numeric section (most specific)
    sec_tokens = []
    for c in uniq:
        s = _looks_like_section_number(c)
        if s:
            sec_tokens.append(s)
    if sec_tokens:
        # keep the first full section number as a part
        parts.append(sec_tokens[0])
    # 2) Table token
    tab_tokens = []
    for c in uniq:
        t = _looks_like_table_number(c)
        if t:
            tab_tokens.append(t)
    if tab_tokens:
        parts.append(tab_tokens[0])
    # 3) Add first uppercase heading if any
    for c in uniq:
        if _looks_like_upper_heading(c):
            parts.append(c)
            break

    section_path = " > ".join([p for p in parts if p])
    return headers_text, section_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest local documents into BigQuery with embeddings.")
    parser.add_argument("--project", required=True, help="GCP project id")
    parser.add_argument("--dataset", default="calrag", help="BigQuery dataset name")
    parser.add_argument("--table", default="documents", help="BigQuery table name")
    parser.add_argument("--location", default="US", help="BigQuery dataset location")
    parser.add_argument("--docs-dir", required=True, help="Directory containing PDFs/MD/TXT")
    # Allow ingesting specific files within docs-dir, one-at-a-time or multiple
    parser.add_argument("--file", dest="file_list_append", action="append", default=[], help="Specific file path under --docs-dir (can repeat)")
    parser.add_argument("--files", nargs="+", default=None, help="Specific file paths under --docs-dir (space-separated)")
    parser.add_argument("--collection", default="", help="Optional logical collection tag")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Max characters per chunk")
    parser.add_argument("--overlap", type=int, default=150, help="Overlap characters between chunks")
    args = parser.parse_args()

    client = bigquery.Client(project=args.project)
    ensure_dataset(client, args.project, args.dataset, args.location)
    ensure_table(client, args.project, args.dataset, args.table)
    table_id = f"{args.project}.{args.dataset}.{args.table}"

    backend, model = resolve_embedding_backend_and_model()
    embed_fn = create_embedding_function()

    root = Path(args.docs_dir).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        print(f"Docs directory not found: {root}")
        sys.exit(1)

    # Resolve selected files (if provided), else enumerate all
    selected_from_repeat = args.file_list_append or []
    selected_from_group = args.files or []
    selected_all: List[str] = [*selected_from_repeat, *selected_from_group]

    files: List[Path]
    if selected_all:
        files = []
        for fname in selected_all:
            p = Path(fname)
            if not p.is_absolute():
                p = root / fname
            p = p.expanduser().resolve()
            if not p.exists() or not p.is_file():
                print(f"[warn] Skipping missing file: {p}")
                continue
            files.append(p)
    else:
        files = enumerate_files(root)

    if not files:
        print("No .pdf/.md/.markdown/.txt files found.")
        return

    rows_buffer: List[Dict[str, Any]] = []
    BATCH = 256
    total_inserted = 0

    for file_path in files:
        print(f"[ingest] Start file: {file_path}")
        suffix = file_path.suffix.lower()
        source_url = str(file_path)
        title = file_path.stem
        mime_type = "text/plain"
        source_type = "txt"
        page_numbers: List[Optional[int]] = []

        if suffix == ".pdf":
            source_type = "pdf"
            mime_type = "application/pdf"
            texts, page_numbers, title = read_pdf_chunks(file_path)
        else:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
            if suffix in {".md", ".markdown"}:
                source_type = "markdown"
                mime_type = "text/markdown"
                texts = smart_chunk_markdown(text, max_len=args.chunk_size, overlap_chars=args.overlap)
                page_numbers = [None] * len(texts)
            else:
                source_type = "txt"
                mime_type = "text/plain"
                # simple char chunking with overlap
                step = max(1, args.chunk_size - max(0, args.overlap))
                texts = []
                i = 0
                while i < len(text):
                    piece = text[i:i + args.chunk_size].strip()
                    if piece:
                        texts.append(piece)
                    i += step
                page_numbers = [None] * len(texts)

        if not texts:
            continue

        print(f"[ingest] File chunks: {len(texts)}")

        # Build metadata rows first
        metadatas: List[Dict[str, Any]] = []
        from utils import build_section_path  # lightweight import
        section_counters: Dict[str, int] = {}
        for idx, chunk in enumerate(texts):
            # For PDF, derive headers and section_path using heuristics
            if source_type == "pdf":
                headers_text, inferred_section_path = extract_headers_for_pdf_chunk(chunk)
                # Fall back to title if no section path inferred
                section_path = (inferred_section_path or title or "").strip()
                headers_value = headers_text
            else:
                # Non-PDF: use markdown-based path and an explicit headers mirror
                section_path = build_section_path(chunk)
                headers_value = section_path
            page_num = page_numbers[idx] if idx < len(page_numbers) else None
            ctr_key = f"{source_url}\n{section_path}\n{page_num if page_num is not None else ''}"
            local_idx = section_counters.get(ctr_key, 0)
            section_counters[ctr_key] = local_idx + 1
            meta = make_chunk_metadata(
                source_url=source_url,
                source_type=source_type,
                section_path=section_path,
                headers=headers_value,
                page_number=page_num,
                chunk_text=chunk,
                title=title,
                mime_type=mime_type,
                embedding_backend=backend,
                embedding_model=model,
                section_local_index=local_idx,
            )
            if args.collection:
                meta["collection"] = args.collection
            metadatas.append(meta)

        # Compute embeddings in batches
        embeddings: List[List[float]] = []
        total_batches = (len(texts) + BATCH - 1) // BATCH
        for batch_index, i in enumerate(range(0, len(texts), BATCH), start=1):
            batch_texts = texts[i:i + BATCH]
            print(f"[embed] {batch_index}/{total_batches} size={len(batch_texts)}")
            embs = embed_fn(batch_texts)  # returns List[List[float]]
            embeddings.extend(embs)

        # Assemble rows
        for meta, chunk_text, emb in zip(metadatas, texts, embeddings):
            row = {
                "chunk_id": meta["chunk_id"],
                "content": chunk_text,
                "embedding": [float(x) for x in emb],
                "source_url": meta.get("source_url", ""),
                "source_type": meta.get("source_type", ""),
                "section_path": meta.get("section_path", ""),
                "headers": meta.get("headers", ""),
                "page_number": int(meta.get("page_number")) if str(meta.get("page_number")) != "" else None,
                "title": meta.get("title", ""),
                "mime_type": meta.get("mime_type", ""),
                "char_count": int(meta.get("char_count", 0) or 0),
                "word_count": int(meta.get("word_count", 0) or 0),
                "content_preview": meta.get("content_preview", ""),
                "embedding_backend": meta.get("embedding_backend", ""),
                "embedding_model": meta.get("embedding_model", ""),
                "inserted_at": meta.get("inserted_at", None),
            }
            rows_buffer.append(row)

            # Flush in smaller batches to avoid 413 Request Entity Too Large
            # With large embeddings (3072 dims), 100 rows is safer than 1000
            if len(rows_buffer) >= 100:
                errors = client.insert_rows_json(table_id, rows_buffer)
                if errors:
                    print(f"Insertion errors: {errors[:3]}")
                inserted_now = len(rows_buffer)
                total_inserted += inserted_now
                rows_buffer.clear()
                print(f"[bq] inserted total_rows={total_inserted}")

    if rows_buffer:
        errors = client.insert_rows_json(table_id, rows_buffer)
        if errors:
            print(f"Insertion errors: {errors[:3]}")
        total_inserted += len(rows_buffer)
        print(f"[bq] inserted total_rows={total_inserted}")

    print("Ingestion complete.")


if __name__ == "__main__":
    main()


