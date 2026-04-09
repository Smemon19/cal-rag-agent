"""Document registration layer for adaptive ingestion."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from uuid import uuid4

from policy_engine.db import run_query, transaction


@dataclass
class RegisteredDocument:
    document_id: str
    batch_id: str
    title: str
    source_uri: str
    content_hash: str
    raw_content: str


class DocumentIngestor:
    @staticmethod
    def _documents_columns() -> dict[str, str]:
        rows = run_query(
            """
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_schema='public' AND table_name='documents'
            """
        )
        return {str(r["column_name"]): str(r["data_type"]) for r in rows}

    def register_document(
        self,
        *,
        batch_id: str,
        source_uri: str,
        raw_content: str,
        title: str | None = None,
        doc_type: str | None = None,
        version: str | None = None,
        effective_date: date | None = None,
    ) -> RegisteredDocument:
        source_uri = source_uri.strip()
        if not source_uri:
            raise ValueError("source_uri is required")

        content_hash = hashlib.sha256(raw_content.encode("utf-8")).hexdigest()
        document_id = f"doc_{uuid4().hex}"
        resolved_title = title or Path(source_uri).name or document_id
        cols = self._documents_columns()

        row_values: dict[str, object] = {
            "title": resolved_title,
            "version": version,
            "effective_date": effective_date,
            "content_hash": content_hash,
            "raw_content": raw_content,
            "ingest_status": "registered",
        }
        if "batch_id" in cols:
            row_values["batch_id"] = batch_id
        if "doc_type" in cols:
            row_values["doc_type"] = doc_type
        if "document_type" in cols:
            row_values["document_type"] = doc_type
        if "source_uri" in cols:
            row_values["source_uri"] = source_uri
        if "source" in cols:
            row_values["source"] = source_uri
        if cols.get("document_id") not in {"integer", "bigint", "smallint"}:
            row_values["document_id"] = document_id

        insert_cols = [c for c in row_values.keys() if c in cols]
        placeholders = ", ".join(["%s"] * len(insert_cols))
        values = tuple(row_values[c] for c in insert_cols)

        conflict_sql = ""
        if "source_uri" in cols and "content_hash" in cols:
            conflict_sql = """
            ON CONFLICT (source_uri, content_hash)
            WHERE source_uri IS NOT NULL AND content_hash IS NOT NULL
            DO UPDATE SET
                title = EXCLUDED.title,
                version = EXCLUDED.version,
                effective_date = EXCLUDED.effective_date,
                raw_content = EXCLUDED.raw_content
            """
            if "batch_id" in cols:
                conflict_sql += ", batch_id = EXCLUDED.batch_id"
            if "doc_type" in cols:
                conflict_sql += ", doc_type = EXCLUDED.doc_type"
            if "document_type" in cols:
                conflict_sql += ", document_type = EXCLUDED.document_type"
            if "updated_at" in cols:
                conflict_sql += ", updated_at = NOW()"
            if "ingest_status" in cols:
                conflict_sql += ", ingest_status = 'registered'"

        with transaction() as cur:
            cur.execute(
                f"""
                INSERT INTO documents ({", ".join(insert_cols)})
                VALUES ({placeholders})
                {conflict_sql}
                RETURNING document_id
                """,
                values,
            )
            returned = cur.fetchone() or {}
            returned_id = returned.get("document_id")

        final_doc_id = str(returned_id if returned_id is not None else document_id)
        return RegisteredDocument(
            document_id=final_doc_id,
            batch_id=batch_id,
            title=resolved_title,
            source_uri=source_uri,
            content_hash=content_hash,
            raw_content=raw_content,
        )

