"""
Apply a migration file and verify the objects it was supposed to create.

Usage:
    python scripts/apply_migration.py src/policy_badger/engine/migrations/0005_....sql

The whole file runs in one transaction: if any statement fails nothing is left
half-applied. Verification reads the CREATE TABLE / ADD COLUMN statements out of
the file itself and confirms each object exists afterwards, so a migration that
silently did nothing is reported as a failure rather than a success.
"""

import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from policy_badger.engine.db import get_connection  # noqa: E402

_TABLE_RE = re.compile(r"CREATE TABLE IF NOT EXISTS\s+([a-zA-Z0-9_]+)", re.IGNORECASE)
_COLUMN_RE = re.compile(
    r"ALTER TABLE\s+([a-zA-Z0-9_]+)\s+ADD COLUMN IF NOT EXISTS\s+([a-zA-Z0-9_]+)",
    re.IGNORECASE,
)


def expected_objects(sql: str) -> tuple[set[str], set[tuple[str, str]]]:
    return set(_TABLE_RE.findall(sql)), set(_COLUMN_RE.findall(sql))


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/apply_migration.py <path_to_sql_file>")
        sys.exit(1)

    sql_file = Path(sys.argv[1])
    if not sql_file.exists():
        print(f"File not found: {sql_file}")
        sys.exit(1)

    load_dotenv(ROOT_DIR / ".env", override=True)
    sql = sql_file.read_text(encoding="utf-8")
    tables, columns = expected_objects(sql)

    print(f"Applying migration: {sql_file.name}")
    print(f"  target host: {os.environ.get('DB_HOST', '(unset)')}")
    print(f"  database:    {os.environ.get('DB_NAME', '(unset)')}")
    print(f"  expects {len(tables)} table(s), {len(columns)} added column(s)")

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql)

            missing: list[str] = []

            for table in sorted(tables):
                cur.execute("SELECT to_regclass(%s)", (f"public.{table}",))
                row = cur.fetchone()
                if row and row[0]:
                    print(f"  [ok] table {table}")
                else:
                    missing.append(f"table {table}")
                    print(f"  [MISSING] table {table}")

            for table, column in sorted(columns):
                cur.execute(
                    """
                    SELECT 1 FROM information_schema.columns
                    WHERE table_schema = 'public' AND table_name = %s AND column_name = %s
                    """,
                    (table, column),
                )
                if cur.fetchone():
                    print(f"  [ok] column {table}.{column}")
                else:
                    # A guarded ALTER is skipped when its table does not exist,
                    # which is expected rather than a failure.
                    cur.execute("SELECT to_regclass(%s)", (f"public.{table}",))
                    parent = cur.fetchone()
                    if parent and parent[0]:
                        missing.append(f"column {table}.{column}")
                        print(f"  [MISSING] column {table}.{column}")
                    else:
                        print(f"  [skipped] column {table}.{column} (table absent)")

            if missing:
                conn.rollback()
                print("\nVerification failed; rolled back. Missing: " + ", ".join(missing))
                sys.exit(1)

        conn.commit()
        print("\nMigration applied and verified.")
    except Exception as e:
        conn.rollback()
        print(f"\nFailed to apply migration (rolled back): {e}")
        sys.exit(1)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
