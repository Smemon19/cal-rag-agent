"""
Read access to the audit trails.

user_audit_log, policy_admin_audit and admin_policy_publish_audit were all
write-only: the rows were recorded but no endpoint or screen ever read them.
These queries expose a single merged, filterable timeline for super admins.
"""

from __future__ import annotations

from typing import Any, Optional

from policy_badger.engine.db import run_query

MAX_PAGE_SIZE = 200

# Audit streams that can be requested individually.
VALID_SOURCES = {"user", "policy", "publish"}


def list_audit_events(
    *,
    source: Optional[str] = None,
    action: Optional[str] = None,
    actor: Optional[str] = None,
    page: int = 1,
    page_size: int = 50,
) -> dict[str, Any]:
    """
    Merged audit timeline, newest first.

    The three underlying tables have different shapes, so each is projected onto
    a common (source, action, actor, subject, details, created_at) tuple before
    being unioned.
    """
    page = max(1, int(page))
    page_size = max(1, min(MAX_PAGE_SIZE, int(page_size)))
    offset = (page - 1) * page_size

    streams: list[str] = []
    if source in (None, "user"):
        streams.append(
            """
            SELECT 'user' AS source,
                   ual.action AS action,
                   actor.username AS actor,
                   target.username AS subject,
                   ual.details AS details,
                   ual.created_at AS created_at
            FROM user_audit_log ual
            LEFT JOIN users actor ON actor.id = ual.actor_user_id
            LEFT JOIN users target ON target.id = ual.target_user_id
            """
        )
    if source in (None, "policy"):
        streams.append(
            """
            SELECT 'policy' AS source,
                   paa.action AS action,
                   actor.username AS actor,
                   paa.policy_id::text AS subject,
                   paa.details AS details,
                   paa.created_at AS created_at
            FROM policy_admin_audit paa
            LEFT JOIN users actor ON actor.id = paa.actor_user_id
            """
        )
    if source in (None, "publish"):
        streams.append(
            """
            SELECT 'publish' AS source,
                   'publish_policy' AS action,
                   apa.published_by AS actor,
                   apa.published_policy_id::text AS subject,
                   jsonb_build_object(
                       'submission_id', apa.submission_id::text,
                       'source_type', apa.source_type,
                       'version', apa.version
                   ) AS details,
                   apa.published_at AS created_at
            FROM admin_policy_publish_audit apa
            """
        )

    if not streams:
        return {"events": [], "total": 0, "page": page, "page_size": page_size, "pages": 1}

    union_sql = "\nUNION ALL\n".join(streams)

    where: list[str] = []
    params: list[Any] = []
    if action:
        where.append("action = %s")
        params.append(action)
    if actor:
        where.append("actor ILIKE %s")
        params.append(f"%{actor}%")
    where_sql = f"WHERE {' AND '.join(where)}" if where else ""

    total_rows = run_query(
        f"SELECT COUNT(*) AS n FROM ({union_sql}) AS merged {where_sql}",
        tuple(params),
    )
    total = int(total_rows[0]["n"]) if total_rows else 0

    events = run_query(
        f"""
        SELECT * FROM ({union_sql}) AS merged
        {where_sql}
        ORDER BY created_at DESC
        LIMIT %s OFFSET %s
        """,
        tuple(params) + (page_size, offset),
    )

    return {
        "events": events,
        "total": total,
        "page": page,
        "page_size": page_size,
        "pages": max(1, (total + page_size - 1) // page_size),
    }


def list_audit_actions() -> list[str]:
    """Distinct action names, for the filter dropdown."""
    rows = run_query(
        """
        SELECT DISTINCT action FROM (
            SELECT action FROM user_audit_log
            UNION
            SELECT action FROM policy_admin_audit
            UNION
            SELECT 'publish_policy' AS action
        ) AS actions
        WHERE action IS NOT NULL
        ORDER BY action
        """
    )
    return [str(r["action"]) for r in rows]
