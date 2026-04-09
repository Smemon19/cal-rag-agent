"""End-to-end adaptive ingestion orchestration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import unquote
from uuid import uuid4

from adaptive_ingestion.document_ingestor import DocumentIngestor
from adaptive_ingestion.foundation import ensure_foundation_schema
from adaptive_ingestion.gap_detector import GapDetector
from adaptive_ingestion.migration_applier import MigrationApplier
from adaptive_ingestion.migration_generator import MigrationGenerator
from adaptive_ingestion.policy_extractor import PolicyExtractor
from adaptive_ingestion.publisher import NonPublishableChunk, Publisher
from adaptive_ingestion.schema_dictionary import SchemaDictionary
from adaptive_ingestion.schema_introspector import SchemaIntrospector
from adaptive_ingestion.schema_planner import SchemaPlanner
from adaptive_ingestion.section_chunker import SectionChunker
from adaptive_ingestion.staging_manager import StagingManager
from policy_engine.db import execute, run_query


@dataclass
class IngestionDocumentInput:
    source_uri: str
    content: str = ""
    title: str | None = None
    doc_type: str | None = None
    version: str | None = None


class AdaptiveIngestionPipeline:
    def __init__(self):
        self.dictionary = SchemaDictionary()
        self.dictionary.refresh_overrides()
        self.doc_ingestor = DocumentIngestor()
        self.chunker = SectionChunker()
        self.extractor = PolicyExtractor(self.dictionary)
        self.staging = StagingManager()
        self.introspector = SchemaIntrospector()
        self.gap_detector = GapDetector(self.dictionary)
        self.schema_planner = SchemaPlanner()
        self.migration_generator = MigrationGenerator()
        self.migration_applier = MigrationApplier()
        self.publisher = Publisher()
        self.bq_mirror = None

    def _get_bq_mirror(self):
        from adaptive_ingestion.bigquery_mirror import BigQueryMirror

        if self.bq_mirror is None:
            self.bq_mirror = BigQueryMirror()
        return self.bq_mirror

    @staticmethod
    def _file_path_from_uri(source_uri: str) -> Path | None:
        if source_uri.startswith("file://"):
            # file:///tmp/a.pdf -> /tmp/a.pdf
            parsed = source_uri[len("file://") :]
            return Path(unquote(parsed))
        candidate = Path(source_uri)
        if candidate.exists():
            return candidate
        return None

    def _extract_content_for_input(self, doc: IngestionDocumentInput) -> str:
        """Return text content, including OCR path for PDF inputs."""
        if doc.content.strip():
            return doc.content

        file_path = self._file_path_from_uri(doc.source_uri)
        if not file_path or not file_path.exists():
            raise ValueError(f"No content provided and source file not found: {doc.source_uri}")

        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            from pdf_loader.pdf_loader import process_pdf

            with TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                output_json = tmp_path / "chunks.json"
                image_dir = tmp_path / "images"
                chunks = process_pdf(file_path, output_json, image_dir, chunk_size=500)
                text = "\n\n".join(str(c.get("text") or "").strip() for c in chunks if str(c.get("text") or "").strip())
                if not text.strip():
                    raise ValueError(f"PDF extraction returned no text: {file_path}")
                return text

        if suffix == ".docx":
            try:
                from docx import Document
            except Exception as e:  # pragma: no cover - dependency/runtime guard
                raise RuntimeError("python-docx is required to ingest .docx files") from e

            docx_obj = Document(str(file_path))
            paragraphs = [p.text.strip() for p in docx_obj.paragraphs if (p.text or "").strip()]
            if not paragraphs:
                raise ValueError(f"DOCX extraction returned no text: {file_path}")
            return "\n\n".join(paragraphs)

        return file_path.read_text(encoding="utf-8")

    def create_batch(self, *, triggered_by: str, trigger_source: str = "manual") -> str:
        batch_id = f"bat_{uuid4().hex}"
        execute(
            """
            INSERT INTO ingestion_batches (batch_id, triggered_by, trigger_source, status, started_at)
            VALUES (%s, %s, %s, 'running', NOW())
            """,
            (batch_id, triggered_by, trigger_source),
        )
        return batch_id

    def run_batch(self, *, batch_id: str, documents: list[IngestionDocumentInput]) -> dict:
        ensure_foundation_schema()
        doc_count = 0
        section_count = 0
        extract_count = 0
        for doc in documents:
            resolved_content = self._extract_content_for_input(doc)
            reg = self.doc_ingestor.register_document(
                batch_id=batch_id,
                source_uri=doc.source_uri,
                raw_content=resolved_content,
                title=doc.title,
                doc_type=doc.doc_type,
                version=doc.version,
            )
            doc_count += 1
            sections = self.chunker.persist_sections(document_id=reg.document_id, text=resolved_content)
            section_count += len(sections)
            for sec in sections:
                ext = self.extractor.extract(
                    document_id=reg.document_id,
                    section_id=sec.section_id,
                    section_text=sec.section_text,
                )
                self.staging.stage_payload(
                    batch_id=batch_id,
                    document_id=reg.document_id,
                    section_id=sec.section_id,
                    payload=ext.payload,
                    confidence=ext.confidence,
                )
                extract_count += 1

        execute(
            """
            UPDATE ingestion_batches
            SET documents_total = %s, documents_processed = %s, sections_total = %s, extractions_total = %s
            WHERE batch_id = %s
            """,
            (doc_count, doc_count, section_count, extract_count, batch_id),
        )
        return {"batch_id": batch_id, "documents_total": doc_count, "sections_total": section_count, "extractions_total": extract_count}

    def plan_schema_changes(self, *, batch_id: str) -> list[dict]:
        staged = self.staging.pending_for_batch(batch_id)
        cols = set(self.introspector.list_table_columns("policies_v2").keys())
        gaps = self.gap_detector.detect(staged, cols)
        proposals = self.schema_planner.build_proposals(batch_id=batch_id, gaps=gaps)
        self.schema_planner.persist_proposals(proposals)
        return [
            {"request_id": p.request_id, "batch_id": p.batch_id, "migration_class": p.migration_class, "items": p.items}
            for p in proposals
        ]

    def apply_approved_migrations(self, *, applied_by: str = "system") -> list[str]:
        rows = run_query(
            """
            SELECT request_id, proposal_json, migration_class, decision_status
            FROM schema_change_requests
            WHERE decision_status IN ('approved', 'pending')
            ORDER BY created_at ASC
            """
        )
        migration_ids = []
        for row in rows:
            sqls = self.migration_generator.generate(row)
            if not sqls and row.get("migration_class") != "safe_auto":
                continue
            migration_ids.append(self.migration_applier.apply(request_row=row, sql_statements=sqls, applied_by=applied_by))
        return migration_ids

    def replay_and_publish(self, *, batch_id: str) -> dict:
        staged = self.staging.pending_for_batch(batch_id)
        published = 0
        event_rows = []
        current_rows = []
        for row in staged:
            try:
                policy_id = self.publisher.publish_row(row)
                self.staging.mark_status(str(row["extraction_id"]), "published")
                published += 1
                event_rows.append(
                    {
                        "event_id": f"evt_{uuid4().hex}",
                        "policy_id": policy_id,
                        "batch_id": batch_id,
                        "event_type": "publish",
                        "payload": {"extraction_id": row["extraction_id"]},
                    }
                )
                current_rows.append(
                    {
                        "policy_id": policy_id,
                        "payload": {"batch_id": batch_id, "extraction_id": row["extraction_id"]},
                    }
                )
            except NonPublishableChunk as e:
                self.staging.mark_status(str(row["extraction_id"]), "skipped_non_publishable", str(e))
            except Exception as e:
                self.staging.mark_status(str(row["extraction_id"]), "failed_publish", str(e))

        try:
            bq_mirror = self._get_bq_mirror()
            bq_mirror.mirror_events(event_rows)
            bq_mirror.upsert_current(current_rows)
        except Exception:
            pass

        cumulative_counts = run_query(
            """
            SELECT
                COALESCE(SUM(CASE WHEN status = 'published' THEN 1 ELSE 0 END), 0) AS published_count,
                COALESCE(SUM(CASE WHEN status = 'failed_publish' THEN 1 ELSE 0 END), 0) AS failed_count,
                COALESCE(SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END), 0) AS pending_count,
                COALESCE(SUM(CASE WHEN status = 'skipped_non_publishable' THEN 1 ELSE 0 END), 0) AS skipped_count
            FROM policy_extractions_staging
            WHERE batch_id = %s
            """,
            (batch_id,),
        )[0]

        summary = {
            "published": int(cumulative_counts.get("published_count") or 0),
            "failed_publish": int(cumulative_counts.get("failed_count") or 0),
            "pending": int(cumulative_counts.get("pending_count") or 0),
            "skipped_non_publishable": int(cumulative_counts.get("skipped_count") or 0),
            "last_run_published": int(published),
        }

        execute(
            """
            UPDATE ingestion_batches
            SET status = %s, finished_at = NOW(), result_summary = %s
            WHERE batch_id = %s
            """,
            ("completed", json.dumps(summary), batch_id),
        )
        return {"batch_id": batch_id, "published": published, "summary": summary}

