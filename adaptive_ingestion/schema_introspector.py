"""Database schema introspection utilities."""

from __future__ import annotations

from typing import Any

from policy_engine.db import run_query


class SchemaIntrospector:
    def list_table_columns(self, table_name: str) -> dict[str, str]:
        rows = run_query(
            """
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = %s
            """,
            (table_name,),
        )
        return {str(r["column_name"]): str(r["data_type"]) for r in rows}

    def list_tables(self) -> list[str]:
        rows = run_query(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            ORDER BY table_name
            """
        )
        return [str(r["table_name"]) for r in rows]

    def snapshot(self) -> dict[str, Any]:
        tables = self.list_tables()
        return {
            "tables": tables,
            "columns": {t: self.list_table_columns(t) for t in tables},
        }

