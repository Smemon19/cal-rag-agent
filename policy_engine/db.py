"""
PostgreSQL access for policy_engine using psycopg2.
Loads credentials from environment (.env via python-dotenv).

Install driver if needed: pip install psycopg2-binary
"""

import os
from typing import Any
from contextlib import contextmanager

from dotenv import load_dotenv
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
except Exception:  # pragma: no cover
    psycopg2 = None  # type: ignore
    RealDictCursor = None  # type: ignore

# Load .env when this module is imported (idempotent). override=True so .env beats stale shell exports.
load_dotenv(override=True)


def get_connection():
    """
    Open a new connection to Cloud SQL / Postgres using DB_* env vars.
    Caller is responsible for closing (or use run_query which closes internally).
    """
    if psycopg2 is None:
        raise RuntimeError("psycopg2 is required for policy_engine.db. Install psycopg2-binary.")

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


def execute(sql: str, params: tuple[Any, ...] | None = None) -> int:
    """Execute a write statement and return affected row count."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params or ())
            rowcount = cur.rowcount
        conn.commit()
        return int(rowcount or 0)
    finally:
        conn.close()


def execute_many(sql: str, params_list: list[tuple[Any, ...]]) -> int:
    """Execute a statement for multiple parameter tuples."""
    if not params_list:
        return 0
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.executemany(sql, params_list)
            rowcount = cur.rowcount
        conn.commit()
        return int(rowcount or 0)
    finally:
        conn.close()


@contextmanager
def transaction():
    """Yield a cursor inside a managed transaction."""
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            yield cur
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
