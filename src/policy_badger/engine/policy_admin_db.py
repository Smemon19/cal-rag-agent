"""
Published-policy management.

Publishing was one-way: rows landed in policies_v2 and nothing could list,
correct, supersede or retire them afterwards. These helpers add that lifecycle.

policies_v2 grows columns over time via the adaptive-ingestion schema evolution
path, so every column referenced here is intersected with a live
information_schema lookup before it reaches SQL. Identifiers are additionally
allowlisted, never interpolated from caller input.

policy_id is an integer on the current deployment but text in
policy_publish_audit, so lookups compare policy_id::text, the same approach
query_builder.py takes. That keeps this module correct whichever type the
column happens to have.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from fastapi import HTTPException

from policy_badger.engine.db import execute, run_query, transaction

POLICY_TABLE = "policies_v2"

# Columns shown in the library list view, when present on the table.
LIST_COLUMNS = [
    "policy_id", "topic", "subtopic", "policy_category", "condition_text",
    "action_text", "source_quote", "status", "updated_at",
]

# Columns an admin may edit through the UI. Deliberately excludes policy_id,
# section_id and the lifecycle bookkeeping columns.
EDITABLE_COLUMNS = {
    "topic", "subtopic", "policy_category", "condition_text", "action_text",
    "source_quote", "exception_text", "recommendation_text", "approver",
    "approval_required", "bill_to", "charge_code",
}

# Free-text search targets.
SEARCH_COLUMNS = [
    "topic", "subtopic", "policy_category", "condition_text",
    "action_text", "source_quote",
]

VALID_STATUSES = {"active", "archived", "superseded"}

MAX_PAGE_SIZE = 100


def _table_columns() -> set[str]:
    rows = run_query(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = %s
        """,
        (POLICY_TABLE,),
    )
    return {str(r["column_name"]) for r in rows}


def _quote_ident(name: str) -> str:
    """Quote an identifier that has already been checked against live columns."""
    if not name.replace("_", "").isalnum():
        raise ValueError(f"Invalid column name: {name!r}")
    return f'"{name}"'


def _log(actor_user_id: Optional[str], policy_id: str, action: str, details: dict) -> None:
    execute(
        """
        INSERT INTO policy_admin_audit (actor_user_id, policy_id, action, details)
        VALUES (%s, %s, %s, %s)
        """,
        (actor_user_id, policy_id, action, json.dumps(details, default=str)),
    )


def list_policies(
    *,
    search: Optional[str] = None,
    status: Optional[str] = None,
    topic: Optional[str] = None,
    page: int = 1,
    page_size: int = 25,
) -> dict[str, Any]:
    """Paginated, filterable view of published policies."""
    available = _table_columns()
    if not available:
        raise HTTPException(status_code=503, detail="Policy table is not available.")

    select_cols = [c for c in LIST_COLUMNS if c in available]
    if "policy_id" not in select_cols:
        raise HTTPException(status_code=503, detail="Policy table is missing policy_id.")

    where: list[str] = []
    params: list[Any] = []

    if status:
        if status not in VALID_STATUSES:
            raise HTTPException(status_code=400, detail="Unknown status filter.")
        if "status" in available:
            where.append("status = %s")
            params.append(status)
    elif "status" in available:
        # Default view hides retired rows.
        where.append("COALESCE(status, 'active') <> 'archived'")

    if topic and "topic" in available:
        where.append("topic = %s")
        params.append(topic)

    if search:
        targets = [c for c in SEARCH_COLUMNS if c in available]
        if targets:
            clause = " OR ".join(f"COALESCE({_quote_ident(c)}, '') ILIKE %s" for c in targets)
            where.append(f"({clause})")
            params.extend([f"%{search}%"] * len(targets))

    where_sql = f"WHERE {' AND '.join(where)}" if where else ""

    total_rows = run_query(f"SELECT COUNT(*) AS n FROM {POLICY_TABLE} {where_sql}", tuple(params))
    total = int(total_rows[0]["n"]) if total_rows else 0

    page = max(1, int(page))
    page_size = max(1, min(MAX_PAGE_SIZE, int(page_size)))
    offset = (page - 1) * page_size

    order_sql = "ORDER BY updated_at DESC" if "updated_at" in available else "ORDER BY policy_id"
    columns_sql = ", ".join(_quote_ident(c) for c in select_cols)

    rows = run_query(
        f"SELECT {columns_sql} FROM {POLICY_TABLE} {where_sql} {order_sql} LIMIT %s OFFSET %s",
        tuple(params) + (page_size, offset),
    )

    return {
        "policies": rows,
        "total": total,
        "page": page,
        "page_size": page_size,
        "pages": max(1, (total + page_size - 1) // page_size),
    }


def list_topics() -> list[str]:
    """Distinct topics, for the library's filter dropdown and the sidebar rail."""
    if "topic" not in _table_columns():
        return []
    rows = run_query(
        f"""
        SELECT DISTINCT topic
        FROM {POLICY_TABLE}
        WHERE topic IS NOT NULL AND topic <> ''
        ORDER BY topic
        LIMIT 100
        """
    )
    return [str(r["topic"]) for r in rows]


def _as_id(policy_id: Any) -> str:
    return str(policy_id).strip()


def get_policy(policy_id: str) -> dict[str, Any]:
    """Every column of one policy, for the edit form."""
    available = _table_columns()
    if not available:
        raise HTTPException(status_code=503, detail="Policy table is not available.")
    columns_sql = ", ".join(_quote_ident(c) for c in sorted(available))
    rows = run_query(
        f"SELECT {columns_sql} FROM {POLICY_TABLE} WHERE policy_id::text = %s",
        (_as_id(policy_id),),
    )
    if not rows:
        raise HTTPException(status_code=404, detail="Policy not found.")
    return rows[0]


def editable_fields() -> list[str]:
    """Editable columns that actually exist on this deployment's table."""
    return sorted(EDITABLE_COLUMNS & _table_columns())


def update_policy(actor_user_id: str, policy_id: str, updates: dict[str, Any]) -> dict[str, Any]:
    """Apply an admin correction to a published policy."""
    policy_id = _as_id(policy_id)
    allowed = EDITABLE_COLUMNS & _table_columns()
    clean = {k: v for k, v in (updates or {}).items() if k in allowed}
    if not clean:
        raise HTTPException(status_code=400, detail="No editable fields supplied.")

    before = get_policy(policy_id)

    set_sql = ", ".join(f"{_quote_ident(k)} = %s" for k in clean)
    params = tuple(clean.values()) + (policy_id,)
    changed = execute(f"UPDATE {POLICY_TABLE} SET {set_sql} WHERE policy_id::text = %s", params)
    if not changed:
        raise HTTPException(status_code=404, detail="Policy not found.")

    _log(
        actor_user_id,
        policy_id,
        "edit_policy",
        {
            "fields": sorted(clean.keys()),
            "before": {k: before.get(k) for k in clean},
            "after": clean,
        },
    )
    return get_policy(policy_id)


def archive_policy(actor_user_id: str, policy_id: str, reason: str = "") -> None:
    """Retire a policy so it stops being retrieved for answers."""
    policy_id = _as_id(policy_id)
    available = _table_columns()
    if "status" not in available:
        raise HTTPException(status_code=503, detail="Policy table has no status column; run migration 0005.")

    get_policy(policy_id)

    sets = ["status = 'archived'"]
    params: list[Any] = []
    if "archived_at" in available:
        sets.append("archived_at = now()")
    if "archived_by_user_id" in available:
        sets.append("archived_by_user_id = %s")
        params.append(actor_user_id)

    execute(
        f"UPDATE {POLICY_TABLE} SET {', '.join(sets)} WHERE policy_id::text = %s",
        tuple(params) + (policy_id,),
    )
    _log(actor_user_id, policy_id, "archive_policy", {"reason": reason})


def restore_policy(actor_user_id: str, policy_id: str) -> None:
    """Return an archived policy to active retrieval."""
    policy_id = _as_id(policy_id)
    available = _table_columns()
    if "status" not in available:
        raise HTTPException(status_code=503, detail="Policy table has no status column; run migration 0005.")

    get_policy(policy_id)

    sets = ["status = 'active'"]
    if "archived_at" in available:
        sets.append("archived_at = NULL")
    if "archived_by_user_id" in available:
        sets.append("archived_by_user_id = NULL")

    execute(f"UPDATE {POLICY_TABLE} SET {', '.join(sets)} WHERE policy_id::text = %s", (policy_id,))
    _log(actor_user_id, policy_id, "restore_policy", {})


def supersede_policy(actor_user_id: str, policy_id: str, replacement_policy_id: str) -> None:
    """
    Mark a policy as replaced by a newer one.

    Both rows must exist, and a policy cannot supersede itself.
    """
    policy_id = _as_id(policy_id)
    replacement_policy_id = _as_id(replacement_policy_id)
    available = _table_columns()
    if "superseded_by_policy_id" not in available or "status" not in available:
        raise HTTPException(status_code=503, detail="Policy table is missing lifecycle columns; run migration 0005.")

    if str(policy_id) == str(replacement_policy_id):
        raise HTTPException(status_code=400, detail="A policy cannot supersede itself.")

    get_policy(policy_id)
    get_policy(replacement_policy_id)

    with transaction() as cur:
        cur.execute(
            f"""
            UPDATE {POLICY_TABLE}
            SET status = 'superseded', superseded_by_policy_id = %s
            WHERE policy_id::text = %s
            """,
            (replacement_policy_id, policy_id),
        )

    _log(
        actor_user_id,
        policy_id,
        "supersede_policy",
        {"superseded_by": str(replacement_policy_id)},
    )
