"""Staging persistence and replay support."""

from __future__ import annotations

import json
from uuid import uuid4

from adaptive_ingestion.contracts import StagingPayload
from policy_engine.db import execute, run_query


class StagingManager:
    def stage_payload(
        self,
        *,
        batch_id: str,
        document_id: str,
        section_id: str,
        payload: StagingPayload,
        confidence: float,
    ) -> str:
        extraction_id = f"ext_{uuid4().hex}"
        execute(
            """
            INSERT INTO policy_extractions_staging (
                extraction_id, batch_id, document_id, section_id,
                candidate_json, mapped_fields_json, unmapped_concepts_json,
                confidence, status
            )
            VALUES (%s, %s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s, 'pending')
            """,
            (
                extraction_id,
                batch_id,
                document_id,
                section_id,
                payload.candidate_json.model_dump_json(),
                json.dumps({k: v.model_dump() for k, v in payload.mapped_fields_json.items()}),
                json.dumps([x.model_dump() for x in payload.unmapped_concepts_json]),
                confidence,
            ),
        )
        return extraction_id

    def pending_for_batch(self, batch_id: str) -> list[dict]:
        return run_query(
            """
            SELECT * FROM policy_extractions_staging
            WHERE batch_id = %s AND status IN ('pending', 'blocked_schema')
            ORDER BY created_at ASC
            """,
            (batch_id,),
        )

    def mark_status(self, extraction_id: str, status: str, publish_error: str | None = None) -> None:
        execute(
            """
            UPDATE policy_extractions_staging
            SET status = %s, publish_error = %s, updated_at = NOW()
            WHERE extraction_id = %s
            """,
            (status, publish_error, extraction_id),
        )

