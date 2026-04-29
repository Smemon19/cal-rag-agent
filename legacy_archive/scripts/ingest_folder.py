#!/usr/bin/env python3
"""Bulk-ingest a folder of PDFs, DOCX, and TXT files into BigQuery vectors."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Ensure repository root is importable when running as: python scripts/ingest_folder.py ...
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv
from docx import Document

from pdf_loader.pdf_text import extract_text
from utils import (
    create_embedding_function,
    get_env_file_path,
    make_chunk_metadata,
    resolve_embedding_backend_and_model,
)
from utils_bigquery import get_bq_dataset, get_bq_project, get_bq_table, insert_rows_json
from utils_vectorstore import get_vector_store


SUPPORTED_SUFFIXES = {".pdf", ".docx", ".txt"}
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
    """Normalize noisy extraction output before chunking."""
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
        if re.match(r"(?i)^page\s+\d+\s*(\|\s*\d+)?$", line):
            continue
        if re.match(r"^[\d\s\|/:.-]+$", line):
            continue
        if len(line.split()) < 4:
            continue
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()


def _read_pdf_text(path: Path) -> Dict[int, str]:
    return extract_text(path, include_tables=True)


def _read_docx_text(path: Path) -> Dict[int, str]:
    doc = Document(path)
    text = "\n".join(p.text for p in doc.paragraphs if (p.text or "").strip())
    return {1: text} if text.strip() else {}


def _read_txt_text(path: Path) -> Dict[int, str]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return {1: text} if text.strip() else {}


def _build_chunks_for_file(
    file_path: Path,
    folder_root: Path,
    page_texts: Dict[int, str],
    chunk_words: int,
    overlap_words: int,
) -> Tuple[List[str], List[str], List[Dict]]:
    backend, emb_model = resolve_embedding_backend_and_model()
    source_url = f"file://{file_path.resolve().as_posix()}"
    title = file_path.stem
    source_type = "pdf" if file_path.suffix.lower() == ".pdf" else ("docx" if file_path.suffix.lower() == ".docx" else "txt")
    mime_type = (
        "application/pdf"
        if source_type == "pdf"
        else ("application/vnd.openxmlformats-officedocument.wordprocessingml.document" if source_type == "docx" else "text/plain")
    )

    ids: List[str] = []
    docs: List[str] = []
    metas: List[Dict] = []
    relative_file_path = str(file_path.relative_to(folder_root))
    relative_folder_path = str(file_path.parent.relative_to(folder_root))
    if relative_folder_path == ".":
        relative_folder_path = ""

    cleaned_pages = [_clean_extracted_text(page_texts[p]) for p in sorted(page_texts.keys())]
    full_text = "\n\n".join([t for t in cleaned_pages if t]).strip()
    chunks = _word_chunks(full_text, chunk_words, overlap_words)
    for idx, chunk in enumerate(chunks):
        if len(chunk.split()) < MIN_CHUNK_WORDS:
            continue
        meta = make_chunk_metadata(
            source_url=source_url,
            source_type=source_type,
            section_path="Document",
            headers="# Document",
            page_number=None,
            chunk_text=chunk,
            title=title,
            mime_type=mime_type,
            embedding_backend=backend,
            embedding_model=emb_model,
            section_local_index=idx,
        )
        meta["original_filename"] = file_path.name
        meta["relative_file_path"] = relative_file_path
        meta["relative_folder_path"] = relative_folder_path
        ids.append(meta["chunk_id"])
        docs.append(chunk)
        metas.append(meta)

    return ids, docs, metas


def _filter_existing(
    vector_store,
    ids: List[str],
    docs: List[str],
    metas: List[Dict],
) -> Tuple[List[str], List[str], List[Dict]]:
    if not ids:
        return ids, docs, metas
    existing = set(vector_store.get_by_ids(ids).get("ids", []))
    if not existing:
        return ids, docs, metas
    keep = [i for i, cid in enumerate(ids) if cid not in existing]
    return [ids[i] for i in keep], [docs[i] for i in keep], [metas[i] for i in keep]


def _embed_and_insert(
    ids: List[str],
    docs: List[str],
    metas: List[Dict],
    *,
    embed_batch_size: int,
    insert_batch_size: int,
) -> int:
    if not docs:
        return 0

    embed_fn = create_embedding_function()
    rows: List[Dict] = []
    total = len(docs)

    for start in range(0, total, embed_batch_size):
        end = min(start + embed_batch_size, total)
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
                    "metadata": json.dumps(m, ensure_ascii=False),
                }
            )

    for start in range(0, len(rows), insert_batch_size):
        end = min(start + insert_batch_size, len(rows))
        insert_rows_json(rows[start:end])

    return len(rows)


def _read_file_texts(path: Path) -> Dict[int, str]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _read_pdf_text(path)
    if suffix == ".docx":
        return _read_docx_text(path)
    if suffix == ".txt":
        return _read_txt_text(path)
    return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest a folder of PDF/DOCX/TXT files into BigQuery.")
    parser.add_argument("folder_path", help="Folder path containing files to ingest.")
    parser.add_argument("--table", default=None, help="BigQuery table/collection name. Defaults to BQ_TABLE.")
    parser.add_argument("--chunk-words", type=int, default=500, help="Target words per chunk.")
    parser.add_argument("--overlap-words", type=int, default=80, help="Word overlap between chunks.")
    parser.add_argument("--embed-batch-size", type=int, default=16, help="Embedding batch size.")
    parser.add_argument("--insert-batch-size", type=int, default=100, help="Insert batch size.")
    parser.add_argument("--dry-run", action="store_true", help="Print files that would be ingested and exit.")
    args = parser.parse_args()

    _load_env()
    if args.table:
        os.environ["BQ_TABLE"] = args.table
        os.environ["RAG_COLLECTION_NAME"] = args.table

    folder = Path(args.folder_path).expanduser().resolve()
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}")

    table_name = get_bq_table()
    print(f"[ingest_folder] BigQuery target: {get_bq_project()}.{get_bq_dataset()}.{table_name}")

    files = [p for p in sorted(folder.rglob("*")) if p.is_file()]
    total_files = len(files)
    candidate_files = [p for p in files if p.suffix.lower() in SUPPORTED_SUFFIXES]
    inserted_chunks_total = 0

    if args.dry_run:
        print(f"[dry-run] Found {len(candidate_files)} ingestible files (recursive):")
        for p in candidate_files:
            rel = p.relative_to(folder)
            print(f"- {rel}")
        print(f"[dry-run] Total scanned files: {total_files}")
        print(f"[dry-run] Total ingestible files: {len(candidate_files)}")
        return

    vector_store = get_vector_store(backend="bigquery", table=table_name)

    for idx, path in enumerate(files, start=1):
        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            print(f"Skipped: {path.name}")
            continue

        print(f"[{idx}/{total_files}] Ingesting: {path.name} ...", end=" ")

        page_texts = _read_file_texts(path)
        if not page_texts:
            print("done (0 chunks)")
            continue

        ids, docs, metas = _build_chunks_for_file(
            path,
            folder,
            page_texts,
            chunk_words=args.chunk_words,
            overlap_words=args.overlap_words,
        )
        ids, docs, metas = _filter_existing(vector_store, ids, docs, metas)
        inserted_now = _embed_and_insert(
            ids,
            docs,
            metas,
            embed_batch_size=args.embed_batch_size,
            insert_batch_size=args.insert_batch_size,
        )
        inserted_chunks_total += inserted_now
        print(f"done ({inserted_now} chunks)")

    print(f"Total: {total_files} files, {inserted_chunks_total} chunks inserted")


if __name__ == "__main__":
    main()
