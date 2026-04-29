"""Minimal PostgreSQL connection diagnostic for Policy Badger."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

load_dotenv(dotenv_path=REPO_ROOT / ".env", override=True)

from policy_engine.db import RealDictCursor, get_connection  # noqa: E402


def _print_config() -> None:
    print("DB configuration:")
    print(f"  DB_HOST={os.environ.get('DB_HOST', '')}")
    print(f"  DB_PORT={os.environ.get('DB_PORT', '5432')}")
    print(f"  DB_NAME={os.environ.get('DB_NAME', 'cal_policy_db')}")
    print(f"  DB_USER={os.environ.get('DB_USER', 'postgres')}")
    print("  DB_PASSWORD=<hidden>")


def _run_query(cur, label: str, sql: str) -> None:
    print(f"\n{label}")
    try:
        cur.execute(sql)
        rows = cur.fetchall()
        print("  SUCCESS")
        print(f"  Rows returned: {len(rows)}")
        for row in rows:
            print(f"  {dict(row)}")
    except Exception as exc:
        print("  FAILURE")
        print(f"  {type(exc).__name__}: {exc}")


def main() -> int:
    _print_config()

    print("\nConnecting to Postgres...")
    try:
        conn = get_connection()
    except Exception as exc:
        print("  FAILURE")
        print(f"  {type(exc).__name__}: {exc}")
        return 1

    print("  SUCCESS")
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            _run_query(cur, "Query: SELECT 1", "SELECT 1")
            _run_query(cur, "Query: SELECT COUNT(*) FROM policies_v2", "SELECT COUNT(*) FROM policies_v2")
            _run_query(cur, "Query: SELECT * FROM policies_v2 LIMIT 3", "SELECT * FROM policies_v2 LIMIT 3")
    finally:
        conn.close()
        print("\nConnection closed.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
