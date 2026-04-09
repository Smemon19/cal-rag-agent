"""Audit helper for ingestion and publish lifecycle."""

from __future__ import annotations

from uuid import uuid4

from policy_engine.db import execute


class AuditLogger:
    def publish_event(self, *, extraction_id: str, policy_id: str | None, status: str, error_text: str | None = None) -> str:
        publish_id = f"pub_{uuid4().hex}"
        execute(
            """
            INSERT INTO policy_publish_audit (publish_id, extraction_id, policy_id, publish_status, error_text)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (publish_id, extraction_id, policy_id, status, error_text),
        )
        return publish_id

