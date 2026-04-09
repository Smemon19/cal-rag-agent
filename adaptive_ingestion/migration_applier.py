"""Apply approved migration plans with policy gating and audit logging."""

from __future__ import annotations

import json
from uuid import uuid4

from policy_engine.db import execute, transaction


ALLOWED_CLASSES = {"safe_auto", "approval_required"}


class MigrationApplier:
    def apply(self, *, request_row: dict, sql_statements: list[str], applied_by: str = "system") -> str:
        request_id = str(request_row.get("request_id"))
        migration_class = str(request_row.get("migration_class") or "")
        decision_status = str(request_row.get("decision_status") or "")

        if migration_class == "disallowed":
            raise ValueError("disallowed migration cannot be applied")
        if migration_class not in ALLOWED_CLASSES:
            raise ValueError(f"unknown migration class: {migration_class}")
        if migration_class == "approval_required" and decision_status != "approved":
            raise ValueError("approval_required migration must be approved first")
        if migration_class == "safe_auto" and decision_status not in {"approved", "auto_approved", "pending"}:
            raise ValueError("safe_auto migration in invalid status")

        migration_id = f"mig_{uuid4().hex}"
        try:
            with transaction() as cur:
                for stmt in sql_statements:
                    if stmt.strip():
                        cur.execute(stmt)
                cur.execute(
                    """
                    INSERT INTO schema_migration_log (
                        migration_id, request_id, migration_sql_json, applied_by, applied_at, status
                    ) VALUES (%s, %s, %s::jsonb, %s, NOW(), 'applied')
                    """,
                    (migration_id, request_id, json.dumps(sql_statements), applied_by),
                )
                cur.execute(
                    """
                    UPDATE schema_change_requests
                    SET decision_status = %s, decided_at = NOW(), updated_at = NOW()
                    WHERE request_id = %s
                    """,
                    ("auto_approved" if migration_class == "safe_auto" else "approved", request_id),
                )
        except Exception as e:
            execute(
                """
                INSERT INTO schema_migration_log (
                    migration_id, request_id, migration_sql_json, applied_by, status, error_text
                ) VALUES (%s, %s, %s::jsonb, %s, 'failed', %s)
                """,
                (migration_id, request_id, json.dumps(sql_statements), applied_by, str(e)),
            )
            raise
        return migration_id

