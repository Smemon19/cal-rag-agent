#!/usr/bin/env python3
"""Diagnostics for retrieval outputs and chunking quality."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List

# Ensure repository root is importable when running as: python scripts/test_retrieval.py ...
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv

from pdf_loader.pdf_text import extract_text
from rag_agent import RAGDeps, build_retrieval_context
from utils import create_embedding_function, get_env_file_path, resolve_embedding_backend_and_model
from utils_vectorstore import get_vector_store


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


def _source_filename(meta: Dict, fallback: str = "unknown") -> str:
    source_url = str((meta or {}).get("source_url") or "")
    title = str((meta or {}).get("title") or "").strip()
    if source_url.startswith("file://"):
        return Path(source_url.replace("file://", "")).name
    if title:
        return title
    return fallback


def _print_retrieved_chunks(question: str, table_name: str, n_results: int) -> None:
    emb_backend, emb_model = resolve_embedding_backend_and_model()
    print(f"[diag] table={table_name}")
    print(f"[diag] embeddings={emb_backend}/{emb_model}")
    print(f"[diag] question={question}")

    vector_store = get_vector_store(backend="bigquery", table=table_name)
    embed_fn = create_embedding_function()
    query_embedding = embed_fn([question])[0]
    res = vector_store.vector_search(query_embedding, n_results=n_results, filters=None)

    ids = res.get("ids", [[]])[0]
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    print(f"[diag] vector_search_results={len(ids)}")

    for idx, chunk_id in enumerate(ids):
        doc = docs[idx] if idx < len(docs) else ""
        meta = metas[idx] if idx < len(metas) else {}
        distance = dists[idx] if idx < len(dists) else None
        wc = len((doc or "").split())
        source_name = _source_filename(meta, fallback=str(chunk_id))
        print("\n" + "=" * 90)
        print(f"Chunk #{idx + 1}")
        print(f"chunk_id: {chunk_id}")
        print(f"source_filename: {source_name}")
        print(f"distance: {distance}")
        print(f"word_count: {wc}")
        print("full_text:")
        print(doc or "")


def _print_agent_context(question: str, table_name: str, n_results: int) -> None:
    _, emb_model = resolve_embedding_backend_and_model()
    deps = RAGDeps(
        vector_store=get_vector_store(backend="bigquery", table=table_name),
        embedding_model=emb_model,
        collection_name=table_name,
        vector_backend="bigquery",
    )

    context_text, structured_entries, debug_payload = build_retrieval_context(
        question,
        deps,
        n_results=n_results,
        return_debug=True,
    )

    print("\n" + "#" * 90)
    print("build_retrieval_context debug payload")
    print(debug_payload)

    print("\n" + "#" * 90)
    print("STRUCTURED ENTRIES")
    print(f"count={len(structured_entries)}")
    for idx, entry in enumerate(structured_entries, start=1):
        print(f"{idx}. {entry}")

    if structured_entries:
        context_for_agent = "\n\n".join(
            f"Source: {entry.get('code_label', 'Doc')} {entry.get('section', '')}\n"
            f"Content: {entry.get('quote', '')}"
            for entry in structured_entries
        )
        context_mode = "structured_entries"
    elif context_text.strip():
        context_for_agent = context_text
        context_mode = "raw_context_fallback"
    else:
        context_for_agent = ""
        context_mode = "no_context"

    print("\n" + "#" * 90)
    print(f"CONTEXT PASSED TO AGENT (mode={context_mode})")
    print(context_for_agent)


def _compare_chunking(pdf_path: Path, chunk_words: int, overlap_words: int) -> None:
    page_texts = extract_text(pdf_path, include_tables=True)

    per_page_chunks: List[str] = []
    for page_num in sorted(page_texts.keys()):
        per_page_chunks.extend(_word_chunks(page_texts[page_num], chunk_words, overlap_words))

    full_text = "\n\n".join(page_texts[p] for p in sorted(page_texts.keys()))
    full_doc_chunks = _word_chunks(full_text, chunk_words, overlap_words)

    print("\n" + "#" * 90)
    print(f"CHUNKING COMPARISON for: {pdf_path.name}")
    print(f"chunk_words={chunk_words}, overlap_words={overlap_words}")
    print(f"pages_with_text={len(page_texts)}")
    print(f"per_page_chunk_count={len(per_page_chunks)}")
    print(f"full_doc_chunk_count={len(full_doc_chunks)}")
    print("per_page_chunk_word_counts=" + str([len(c.split()) for c in per_page_chunks]))
    print("full_doc_chunk_word_counts=" + str([len(c.split()) for c in full_doc_chunks]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Test retrieval chunks and assembled context.")
    parser.add_argument("question", nargs="?", default="", help="Question to run retrieval against.")
    parser.add_argument("--table", default=None, help="BigQuery table name (defaults to BQ_TABLE env).")
    parser.add_argument("--n-results", type=int, default=5, help="Number of retrieval chunks to inspect.")
    parser.add_argument(
        "--compare-pdf",
        default="~/Downloads/badger_info",
        help="PDF file path or root folder to choose first PDF from for chunking comparison.",
    )
    parser.add_argument("--chunk-words", type=int, default=500, help="Chunk size for comparison.")
    parser.add_argument("--overlap-words", type=int, default=80, help="Chunk overlap for comparison.")
    parser.add_argument(
        "--compare-only",
        action="store_true",
        help="Only run per-page vs full-doc chunking comparison; skip retrieval diagnostics.",
    )
    args = parser.parse_args()

    _load_env()
    table_name = args.table or os.getenv("BQ_TABLE", "documents")

    if not args.compare_only:
        if not args.question.strip():
            raise ValueError("Question is required unless --compare-only is used.")
        _print_retrieved_chunks(args.question, table_name, args.n_results)
        _print_agent_context(args.question, table_name, args.n_results)

    compare_target = Path(args.compare_pdf).expanduser().resolve()
    if compare_target.is_dir():
        pdfs = sorted(compare_target.rglob("*.pdf"))
        if not pdfs:
            raise FileNotFoundError(f"No PDFs found under directory: {compare_target}")
        pdf_path = pdfs[0]
    elif compare_target.is_file() and compare_target.suffix.lower() == ".pdf":
        pdf_path = compare_target
    else:
        raise FileNotFoundError(f"Invalid --compare-pdf target: {compare_target}")

    _compare_chunking(pdf_path, args.chunk_words, args.overlap_words)


if __name__ == "__main__":
    main()
