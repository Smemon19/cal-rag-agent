"""Generate SQL migration plans from approved schema proposals."""

from __future__ import annotations

from typing import Any


TYPE_MAP = {
    "new_scalar": "TEXT",
    "new_relationship": "TEXT",
}


class MigrationGenerator:
    def generate(self, proposal: dict[str, Any]) -> list[str]:
        sqls: list[str] = []
        items = ((proposal or {}).get("proposal_json") or {}).get("items", [])
        if isinstance(items, str):
            return []
        for item in items:
            concept = str(item.get("concept") or "").strip()
            gap_type = str(item.get("gap_type") or "").strip()
            if not concept:
                continue
            if gap_type == "new_scalar":
                sqls.append(f'ALTER TABLE policies_v2 ADD COLUMN IF NOT EXISTS "{concept}" {TYPE_MAP["new_scalar"]};')
            elif gap_type == "new_relationship":
                table = f"policy_{concept}"
                sqls.append(
                    f"""
                    CREATE TABLE IF NOT EXISTS {table} (
                        id TEXT PRIMARY KEY,
                        policy_id TEXT NOT NULL,
                        value_text TEXT NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    );
                    """.strip()
                )
            elif gap_type == "new_value":
                # Value expansion is data-level, no DDL required.
                continue
        return sqls

