"""Foundation schema bootstrap."""

from __future__ import annotations

from pathlib import Path

from policy_badger.engine.db import get_connection


def ensure_foundation_schema() -> None:
    migration_path = (
        Path(__file__).resolve().parents[1]
        / "engine"
        / "migrations"
        / "0001_adaptive_ingestion_foundation.sql"
    )
    sql = migration_path.read_text(encoding="utf-8")
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
