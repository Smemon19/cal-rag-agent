"""Publish staged records into production policy tables."""

from __future__ import annotations

import json

from adaptive_ingestion.audit_logger import AuditLogger
from adaptive_ingestion.contracts import CandidateJson
from policy_engine.db import run_query, transaction


class NonPublishableChunk(Exception):
    """Raised when a staged chunk should not be published to policies_v2."""


class Publisher:
    def __init__(self):
        self.audit = AuditLogger()
        self._policies_columns_cache: dict[str, str] | None = None

    def _policies_columns(self) -> dict[str, str]:
        if self._policies_columns_cache is not None:
            return self._policies_columns_cache
        rows = run_query(
            """
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_schema='public' AND table_name='policies_v2'
            """
        )
        self._policies_columns_cache = {str(r["column_name"]): str(r["data_type"]) for r in rows}
        return self._policies_columns_cache

    @staticmethod
    def _merge_columns(mapped_fields: dict) -> tuple[list[str], list]:
        cols = []
        vals = []
        for key, item in mapped_fields.items():
            cols.append(key)
            vals.append(item.get("value") if isinstance(item, dict) else item)
        return cols, vals

    def publish_row(self, staging_row: dict) -> str:
        extraction_id = str(staging_row["extraction_id"])
        candidate = staging_row.get("candidate_json") or {}
        if isinstance(candidate, str):
            candidate = json.loads(candidate)
        # Strict contract validation before publish.
        try:
            candidate_obj = CandidateJson.model_validate(candidate)
            candidate = candidate_obj.model_dump(mode="json")
        except Exception as e:
            raise NonPublishableChunk(f"invalid candidate_json contract: {e}") from e

        publishable = bool(candidate.get("publishable"))
        confidence = float(candidate.get("confidence") or staging_row.get("confidence") or 0.0)
        has_structure = bool(
            candidate.get("topic")
            or candidate.get("subtopic")
            or candidate.get("condition_text")
            or candidate.get("action_text")
            or candidate.get("recommendation_text")
        )
        if not publishable:
            raise NonPublishableChunk(candidate.get("reason_if_not_publishable") or "chunk flagged non-publishable")
        if confidence < 0.62:
            raise NonPublishableChunk(f"confidence below threshold ({confidence:.2f})")
        if not has_structure:
            raise NonPublishableChunk("missing meaningful structured fields")

        mapped = staging_row.get("mapped_fields_json") or {}
        if isinstance(mapped, str):
            mapped = json.loads(mapped)
        cols, vals = self._merge_columns(mapped)
        table_cols = self._policies_columns()

        # Keep only columns that exist in policies_v2.
        filtered_cols = []
        filtered_vals = []
        for c, v in zip(cols, vals):
            if c in table_cols:
                filtered_cols.append(c)
                filtered_vals.append(v)

        insert_cols = []
        insert_vals = []

        # Legacy policies_v2 uses integer policy_id; do NOT insert explicit text ID.
        policy_id_is_integer = table_cols.get("policy_id") in {"integer", "bigint", "smallint"}
        if not policy_id_is_integer and "policy_id" in table_cols:
            # In non-legacy schema, policy_id may be text; caller should provide one elsewhere if needed.
            # Omit here to allow DB defaults when configured.
            pass

        if "section_id" in table_cols:
            insert_cols.append("section_id")
            insert_vals.append(staging_row["section_id"])
        for c, v in zip(filtered_cols, filtered_vals):
            insert_cols.append(c)
            insert_vals.append(v)

        source_quote = (staging_row.get("candidate_json") or {}).get("source_quote")
        if isinstance(staging_row.get("candidate_json"), str):
            source_quote = (json.loads(staging_row["candidate_json"]) or {}).get("source_quote")
        if "source_quote" in table_cols:
            insert_cols.append("source_quote")
            insert_vals.append(source_quote or "")
        if "status" in table_cols:
            insert_cols.append("status")
            insert_vals.append("active")

        columns_sql = ", ".join([f'"{c}"' for c in insert_cols])
        placeholders = ", ".join(["%s"] * len(insert_vals))
        with transaction() as cur:
            cur.execute(
                f"""
                INSERT INTO policies_v2 ({columns_sql})
                VALUES ({placeholders})
                RETURNING policy_id
                """,
                tuple(insert_vals),
            )
            out = cur.fetchone() or {}
            policy_id = str(out.get("policy_id"))

        self.audit.publish_event(extraction_id=extraction_id, policy_id=policy_id, status="published")
        return policy_id

