#!/usr/bin/env python3
"""Ingest a local PDF into BigQuery vector store using Vertex embeddings."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple


# Ensure repository root is importable when running as: python scripts/ingest_pdf.py ...
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv

from pdf_loader.pdf_text import extract_text
from utils import (
    create_embedding_function,
    get_env_file_path,
    make_chunk_metadata,
    resolve_embedding_backend_and_model,
)
from utils_bigquery import get_bq_dataset, get_bq_project, get_bq_table, insert_rows_json
from utils_vectorstore import get_vector_store

MIN_CHUNK_WORDS = 50


def _load_env() -> None:
    repo_env = REPO_ROOT / ".env"
    if repo_env.exists():
        load_dotenv(dotenv_path=repo_env, override=True)
    else:
        load_dotenv(dotenv_path=get_env_file_path(), override=True)


def _word_chunks(text: str, chunk_words: int, overlap_words: int) -> List[str]:
    words = (text or "").split()
    if not words:
        return []
    if chunk_words <= 0:
        raise ValueError("chunk_words must be > 0")
    overlap_words = max(0, min(overlap_words, chunk_words - 1))
    step = chunk_words - overlap_words
    out: List[str] = []
    i = 0
    while i < len(words):
        out.append(" ".join(words[i : i + chunk_words]))
        i += step
    return out


def _clean_extracted_text(text: str) -> str:
    """Normalize noisy PDF extraction output before chunking."""
    if not text:
        return ""

    # Fix spaced-out words like "A u g u s t" -> "August"
    text = re.sub(r"\b(?:[A-Za-z]\s+){2,}[A-Za-z]\b", lambda m: re.sub(r"\s+", "", m.group(0)), text)
    # Fix spaced-out numbers like "2 0 2 5" -> "2025"
    text = re.sub(r"\b(?:\d\s+){2,}\d\b", lambda m: re.sub(r"\s+", "", m.group(0)), text)
    text = re.sub(r"[ \t]+", " ", text)

    cleaned_lines: List[str] = []
    for raw_line in text.splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line:
            continue
        # Drop bare page counters and similar footer artifacts.
        if re.match(r"(?i)^page\s+\d+\s*(\|\s*\d+)?$", line):
            continue
        if re.match(r"^[\d\s\|/:.-]+$", line):
            continue
        # Drop extremely short lines (likely headers/footers).
        if len(line.split()) < 4:
            continue
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()


def _build_chunks(
    page_texts: Dict[int, str],
    source_path: Path,
    chunk_words: int,
    overlap_words: int,
) -> Tuple[List[str], List[str], List[Dict]]:
    backend, emb_model = resolve_embedding_backend_and_model()
    source_url = f"file://{source_path.resolve().as_posix()}"
    title = source_path.stem

    ids: List[str] = []
    docs: List[str] = []
    metas: List[Dict] = []

    page_numbers = sorted(page_texts.keys())
    print(f"[ingest_pdf] Building full-document chunks from {len(page_numbers)} pages...")
    cleaned_pages = [_clean_extracted_text(page_texts[p]) for p in page_numbers]
    full_text = "\n\n".join([t for t in cleaned_pages if t]).strip()
    chunks = _word_chunks(full_text, chunk_words, overlap_words)
    print(f"[ingest_pdf] full_document_words={len(full_text.split())} chunks={len(chunks)}")

    for idx, chunk in enumerate(chunks):
        if len(chunk.split()) < MIN_CHUNK_WORDS:
            continue
        meta = make_chunk_metadata(
            source_url=source_url,
            source_type="pdf",
            section_path="Document",
            headers="# Document",
            page_number=None,
            chunk_text=chunk,
            title=title,
            mime_type="application/pdf",
            embedding_backend=backend,
            embedding_model=emb_model,
            section_local_index=idx,
        )
        ids.append(meta["chunk_id"])
        docs.append(chunk)
        metas.append(meta)

    return ids, docs, metas


def _filter_existing(ids: List[str], docs: List[str], metas: List[Dict]) -> Tuple[List[str], List[str], List[Dict]]:
    print("[ingest_pdf] Checking for existing chunk IDs in BigQuery...")
    vector_store = get_vector_store(backend="bigquery")
    existing = set(vector_store.get_by_ids(ids).get("ids", []))
    if not existing:
        print("[ingest_pdf] No duplicates found.")
        return ids, docs, metas

    keep = [i for i, cid in enumerate(ids) if cid not in existing]
    print(
        f"[ingest_pdf] Duplicate check: existing={len(existing)} "
        f"new={len(keep)} skipped={len(ids) - len(keep)}"
    )
    return [ids[i] for i in keep], [docs[i] for i in keep], [metas[i] for i in keep]


def _embed_and_insert(
    ids: List[str],
    docs: List[str],
    metas: List[Dict],
    *,
    embed_batch_size: int,
    insert_batch_size: int,
) -> None:
    if not docs:
        print("[ingest_pdf] Nothing to insert after duplicate filtering.")
        return

    print("[ingest_pdf] Creating embedding function...")
    embed_fn = create_embedding_function()
    rows: List[Dict] = []

    total = len(docs)
    for start in range(0, total, embed_batch_size):
        end = min(start + embed_batch_size, total)
        print(f"[ingest_pdf] Embedding batch {start + 1}-{end} / {total}")
        embeddings = embed_fn(docs[start:end])

        for i, emb in enumerate(embeddings):
            m = metas[start + i]
            rows.append(
                {
                    "chunk_id": ids[start + i],
                    "content": docs[start + i],
                    "embedding": emb,
                    "source_url": m.get("source_url"),
                    "source_type": m.get("source_type"),
                    "section_path": m.get("section_path"),
                    "headers": m.get("headers"),
                    "page_number": m.get("page_number") or None,
                    "title": m.get("title"),
                    "mime_type": m.get("mime_type"),
                    "char_count": m.get("char_count"),
                    "word_count": m.get("word_count"),
                    "content_preview": m.get("content_preview"),
                    "embedding_backend": m.get("embedding_backend"),
                    "embedding_model": m.get("embedding_model"),
                    "inserted_at": m.get("inserted_at"),
                    "created_at": m.get("inserted_at"),
                    # BigQuery JSON columns accept JSON-serialized payloads reliably
                    # across client versions.
                    "metadata": json.dumps(m, ensure_ascii=False),
                }
            )

    print(f"[ingest_pdf] Inserting {len(rows)} rows into BigQuery...")
    for start in range(0, len(rows), insert_batch_size):
        end = min(start + insert_batch_size, len(rows))
        print(f"[ingest_pdf] Insert batch {start + 1}-{end} / {len(rows)}")
        insert_rows_json(rows[start:end])


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest a PDF into BigQuery vector store.")
    parser.add_argument("pdf_path", help="Path to the PDF file.")
    parser.add_argument("--chunk-words", type=int, default=500, help="Target words per chunk.")
    parser.add_argument("--overlap-words", type=int, default=80, help="Word overlap between chunks.")
    parser.add_argument("--embed-batch-size", type=int, default=16, help="Embedding batch size.")
    parser.add_argument("--insert-batch-size", type=int, default=100, help="BigQuery insert batch size.")
    args = parser.parse_args()

    _load_env()

    pdf_path = Path(args.pdf_path).expanduser().resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    print(f"[ingest_pdf] PDF: {pdf_path}")
    print(
        f"[ingest_pdf] BigQuery target: {get_bq_project()}.{get_bq_dataset()}.{get_bq_table()}"
    )
    print(
        f"[ingest_pdf] Chunking: chunk_words={args.chunk_words}, overlap_words={args.overlap_words}"
    )

    page_texts = extract_text(pdf_path, include_tables=True)
    print(f"[ingest_pdf] Extracted text from {len(page_texts)} pages.")

    ids, docs, metas = _build_chunks(
        page_texts,
        pdf_path,
        chunk_words=args.chunk_words,
        overlap_words=args.overlap_words,
    )
    print(f"[ingest_pdf] Prepared {len(docs)} chunks before dedup.")

    ids, docs, metas = _filter_existing(ids, docs, metas)
    _embed_and_insert(
        ids,
        docs,
        metas,
        embed_batch_size=args.embed_batch_size,
        insert_batch_size=args.insert_batch_size,
    )
    print("[ingest_pdf] Done.")


if __name__ == "__main__":
    main()
