"""
PostgreSQL access for policy_engine using psycopg2.
Loads credentials from environment (.env via python-dotenv).

Install driver if needed: pip install psycopg2-binary
"""

import os
from typing import Any

import psycopg2
from dotenv import load_dotenv
from psycopg2.extras import RealDictCursor

# Load .env when this module is imported (idempotent). override=True so .env beats stale shell exports.
load_dotenv(override=True)


def get_connection():
    """
    Open a new connection to Cloud SQL / Postgres using DB_* env vars.
    Caller is responsible for closing (or use run_query which closes internally).
    """
    host = (os.environ.get("DB_HOST") or "").strip()
    if not host:
        raise ValueError(
            "DB_HOST is missing or empty. Without a host, psycopg2 uses a local Unix socket "
            "(e.g. /tmp/.s.PGSQL.5432), not Cloud SQL. Set DB_HOST to the instance IP, hostname, "
            "or 127.0.0.1 if using the Cloud SQL Auth Proxy."
        )

    password = os.environ.get("DB_PASSWORD")
    if password is None:
        raise ValueError("DB_PASSWORD is not set in the environment.")

    return psycopg2.connect(
        host=host,
        port=os.environ.get("DB_PORT", "5432"),
        dbname=os.environ.get("DB_NAME", "cal_policy_db"),
        user=os.environ.get("DB_USER", "postgres"),
        password=password,
    )


def run_query(sql: str, params: tuple[Any, ...] | None = None) -> list[dict]:
    """
    Execute a parameterized SELECT (or any query that returns rows).
    Returns rows as list[dict] mapping column name → value.
    Always pass placeholders (%s) and params — never interpolate user input into SQL.
    """
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, params or ())
            fetched = cur.fetchall()
            return [dict(row) for row in fetched]
    finally:
        conn.close()
