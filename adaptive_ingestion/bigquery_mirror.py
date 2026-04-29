"""BigQuery mirroring for append-only events and current-state rows."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

from google.cloud import bigquery


def _get_bq_project() -> str:
    project = (
        os.getenv("BQ_PROJECT")
        or os.getenv("GOOGLE_CLOUD_PROJECT")
        or os.getenv("GCLOUD_PROJECT")
    )
    if project:
        return project
    client = bigquery.Client()
    if not client.project:
        raise RuntimeError("BigQuery project is not configured.")
    return str(client.project)


def _get_bq_dataset() -> str:
    return os.getenv("BQ_DATASET") or os.getenv("BIGQUERY_DATASET") or "cal_policy"


def _get_bq_client(project: str) -> bigquery.Client:
    return bigquery.Client(project=project)


def _insert_rows_json(rows: list[dict[str, Any]], *, project: str, dataset: str, table: str) -> None:
    client = _get_bq_client(project)
    table_ref = f"{project}.{dataset}.{table}"
    errors = client.insert_rows_json(table_ref, rows)
    if errors:
        raise RuntimeError(f"BigQuery insert_rows_json failed for {table_ref}: {errors}")


class BigQueryMirror:
    def __init__(self) -> None:
        self.project = _get_bq_project()
        self.dataset = os.getenv("BQ_POLICY_DATASET", _get_bq_dataset())
        self.events_table = os.getenv("BQ_POLICY_EVENTS_TABLE", "policy_events")
        self.current_table = os.getenv("BQ_POLICY_CURRENT_TABLE", "policies_current")

    def mirror_events(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        payload = []
        now = datetime.now(timezone.utc).isoformat()
        for row in rows:
            payload.append(
                {
                    "event_id": row.get("event_id"),
                    "policy_id": row.get("policy_id"),
                    "batch_id": row.get("batch_id"),
                    "event_type": row.get("event_type", "publish"),
                    "event_time": now,
                    "payload_json": json.dumps(row.get("payload", {})),
                }
            )
        _insert_rows_json(payload, project=self.project, dataset=self.dataset, table=self.events_table)

    def upsert_current(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        client = _get_bq_client(self.project)
        full_table = f"{self.project}.{self.dataset}.{self.current_table}"
        query = f"""
        MERGE `{full_table}` t
        USING (SELECT @policy_id AS policy_id, @payload_json AS payload_json, @updated_at AS updated_at) s
        ON t.policy_id = s.policy_id
        WHEN MATCHED THEN
          UPDATE SET payload_json = s.payload_json, updated_at = s.updated_at
        WHEN NOT MATCHED THEN
          INSERT (policy_id, payload_json, updated_at) VALUES (s.policy_id, s.payload_json, s.updated_at)
        """
        for row in rows:
            policy_id = str(row.get("policy_id") or "")
            if not policy_id:
                continue
            job = client.query(
                query,
                job_config=bigquery.QueryJobConfig(
                    query_parameters=[
                        bigquery.ScalarQueryParameter("policy_id", "STRING", policy_id),
                        bigquery.ScalarQueryParameter("payload_json", "STRING", json.dumps(row.get("payload", {}))),
                        bigquery.ScalarQueryParameter(
                            "updated_at",
                            "TIMESTAMP",
                            datetime.now(timezone.utc).isoformat(),
                        ),
                    ]
                ),
            )
            job.result()

