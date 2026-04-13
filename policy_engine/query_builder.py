"""
Build safe, parameterized SELECT queries from a validated search spec.
Only allowlisted identifiers appear in SQL; values are always bound via %s / ANY(%s).
"""

from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import Any

from policy_engine.planner import ALLOWED_FILTER_KEYS, ALLOWED_REQUESTED_FIELDS

DEFAULT_LIMIT = 20

FIELD_SQL_MAP: dict[str, str] = {
    "document_id": "COALESCE(s.document_id::text, d.document_id::text)",
    "batch_id": "COALESCE(s.batch_id, d.batch_id)",
    "summary": "COALESCE(s.candidate_json->>'summary', p.condition_text, p.action_text)",
    "topic": "COALESCE(NULLIF(p.topic, ''), s.candidate_json->>'topic')",
    "subtopic": "COALESCE(NULLIF(p.subtopic, ''), s.candidate_json->>'subtopic')",
    "condition_text": "COALESCE(NULLIF(p.condition_text, ''), s.candidate_json->>'condition_text')",
    "action_text": "COALESCE(NULLIF(p.action_text, ''), s.candidate_json->>'action_text')",
    "source_quote": "COALESCE(NULLIF(p.source_quote, ''), s.candidate_json->>'source_quote')",
}

FILTER_SQL_MAP: dict[str, str] = {
    "document_id": "COALESCE(s.document_id::text, d.document_id::text)",
    "batch_id": "COALESCE(s.batch_id, d.batch_id)",
    "topic": "COALESCE(NULLIF(p.topic, ''), s.candidate_json->>'topic')",
    "subtopic": "COALESCE(NULLIF(p.subtopic, ''), s.candidate_json->>'subtopic')",
}

BASE_FROM_SQL = """
FROM policies_v2 p
LEFT JOIN policy_publish_audit a
  ON a.policy_id = p.policy_id::text
 AND a.publish_status = 'published'
LEFT JOIN policy_extractions_staging s
  ON s.extraction_id = a.extraction_id
LEFT JOIN sections sec
  ON sec.section_id::text = p.section_id::text
LEFT JOIN documents d
  ON d.document_id::text = sec.document_id::text
"""


def _quote_ident(name: str) -> str:
    """Quote identifier for PostgreSQL (allowlisted names only)."""
    if not name.replace("_", "").isalnum():
        raise ValueError(f"Invalid column name for SQL builder: {name!r}")
    return f'"{name}"'


def _select_expr(name: str) -> str:
    return FIELD_SQL_MAP.get(name, f"p.{_quote_ident(name)}")


def _filter_expr(name: str) -> str:
    return FILTER_SQL_MAP.get(name, f"p.{_quote_ident(name)}")


def build_query_from_spec(search_spec: dict) -> tuple[str, tuple[Any, ...]]:
    """
    Build SELECT ... FROM policies_v2 WHERE ... LIMIT n (default 10).
    - Scalar filters: column = %s
    - List filters: column = ANY(%s) with a Python list bound as one parameter
    """
    table = search_spec.get("table")
    if table != "policies_v2":
        raise ValueError('Only table "policies_v2" is supported.')

    fields: list[str] = search_spec.get("requested_fields") or []
    if not fields:
        raise ValueError("requested_fields must be non-empty (validated upstream).")

    for f in fields:
        if f not in ALLOWED_REQUESTED_FIELDS:
            raise ValueError(f'REQUESTED field not allowlisted: "{f}"')

    select_clause = ", ".join(f'{_select_expr(f)} AS "{f}"' for f in fields)
    filters: dict[str, Any] = dict(search_spec.get("filters") or {})

    clauses: list[str] = []
    params: list[Any] = []

    for col in sorted(filters.keys()):
        if col not in ALLOWED_FILTER_KEYS:
            raise ValueError(f'Unknown filter column "{col}".')
        val = filters[col]
        expr = _filter_expr(col)

        # topic and subtopic use case-insensitive matching so planner
        # capitalisation differences don't cause zero-row misses.
        case_insensitive = col in {"topic", "subtopic"}

        if isinstance(val, list):
            if not val:
                continue
            if col == "policy_category":
                clauses.append(f"({expr} = ANY(%s) OR {expr} IS NULL)")
                params.append(val)
            elif case_insensitive:
                clauses.append(f"LOWER({expr}) = ANY(%s)")
                params.append([v.lower() for v in val])
            else:
                clauses.append(f"{expr} = ANY(%s)")
                params.append(val)
        else:
            if col == "policy_category":
                clauses.append(f"({expr} = %s OR {expr} IS NULL)")
                params.append(val)
            elif case_insensitive:
                clauses.append(f"LOWER({expr}) = LOWER(%s)")
                params.append(val)
            else:
                clauses.append(f"{expr} = %s")
                params.append(val)

    where_sql = " AND ".join(clauses) if clauses else "TRUE"
    sql = (
        f"SELECT {select_clause}\n"
        f"{BASE_FROM_SQL.strip()}\n"
        f"WHERE {where_sql}\n"
        f"ORDER BY\n"
        f"  (CASE WHEN p.action_text IS NOT NULL AND p.condition_text IS NOT NULL THEN 0\n"
        f"        WHEN p.action_text IS NOT NULL THEN 1\n"
        f"        WHEN p.condition_text IS NOT NULL THEN 2\n"
        f"        ELSE 3 END),\n"
        f"  p.policy_id\n"
        f"LIMIT {int(DEFAULT_LIMIT)}"
    )
    return sql, tuple(params)


def relaxed_specs_in_order(search_spec: dict) -> list[dict]:
    """
    Produce broader validated spec variants to try when the primary query returns 0 rows.

    Order:
    1. Same spec but filters without activity_type (if it was present).
    2. Filters collapsed to only policy_category (if present) and status — drops topic, etc.

    Returns [] if no strictly broader variant exists.
    """
    base = deepcopy(search_spec)
    filters: dict[str, Any] = dict(base.get("filters") or {})
    variants: list[dict] = []
    seen_fingerprints: set[str] = set()

    def _add_variant(new_filters: dict[str, Any]) -> None:
        if new_filters == filters:
            return
        fp = json.dumps(new_filters, sort_keys=True, default=str)
        if fp in seen_fingerprints:
            return
        seen_fingerprints.add(fp)
        s = deepcopy(base)
        s["filters"] = new_filters
        variants.append(s)

    # Attempt 1: remove activity_type
    if "activity_type" in filters:
        f1 = {k: v for k, v in filters.items() if k != "activity_type"}
        _add_variant(f1)

    # Attempt 2: only scope + policy_category + status (when current filters are broader than that)
    pc = filters.get("policy_category")
    st = filters.get("status", "active")
    minimal: dict[str, Any] = {"status": st}
    for scoped in ("batch_id", "document_id"):
        if scoped in filters and not _is_empty_value(filters.get(scoped)):
            minimal[scoped] = filters.get(scoped)
    if pc is not None and not _is_empty_value(pc):
        minimal["policy_category"] = pc

    keys_set = set(filters.keys())
    if keys_set <= {"policy_category", "status", "batch_id", "document_id"}:
        pass  # already minimal; no second variant
    elif minimal != filters:
        _add_variant(minimal)

    # Attempt 3: scope + status only (drop policy_category completely).
    scope_status: dict[str, Any] = {"status": st}
    for scoped in ("batch_id", "document_id"):
        if scoped in filters and not _is_empty_value(filters.get(scoped)):
            scope_status[scoped] = filters.get(scoped)
    if scope_status != filters:
        _add_variant(scope_status)

    return variants


def _is_empty_value(val: Any) -> bool:
    if val is None:
        return True
    if isinstance(val, str) and not val.strip():
        return True
    if isinstance(val, list) and len(val) == 0:
        return True
    return False


def build_relaxed_query_from_spec(search_spec: dict) -> tuple[str, tuple[Any, ...]] | None:
    """
    Build SQL for the first applicable relaxed search spec, or None if no relaxation exists.
    Prefer using relaxed_specs_in_order + loop in the orchestrator for multiple fallbacks;
    this wrapper matches the public API for a single fallback attempt.
    """
    specs = relaxed_specs_in_order(search_spec)
    if not specs:
        return None
    return build_query_from_spec(specs[0])


def build_text_fallback_query(search_spec: dict, question: str) -> tuple[str, tuple[Any, ...]] | None:
    """Fallback query using semantic text fields when strict filters return zero rows."""
    q = (question or "").strip().lower()
    if not q:
        return None

    tokens = re.findall(r"[a-z0-9_]+", q)
    stop = {"what", "when", "where", "which", "how", "should", "would", "could", "from", "into", "with", "the", "and", "for", "that", "this", "have", "your"}
    keywords = [t for t in tokens if len(t) >= 4 and t not in stop][:6]
    if not keywords:
        return None

    fields: list[str] = search_spec.get("requested_fields") or ["topic", "subtopic", "summary", "condition_text", "action_text", "source_quote"]
    valid_fields = [f for f in fields if f in ALLOWED_REQUESTED_FIELDS] or ["topic", "subtopic", "summary", "condition_text", "action_text", "source_quote"]
    select_clause = ", ".join(f'{_select_expr(f)} AS "{f}"' for f in valid_fields)

    filters = dict(search_spec.get("filters") or {})
    clauses: list[str] = []
    params: list[Any] = []

    # Keep scope/status constraints for safety and relevance.
    for key in ("status", "batch_id", "document_id"):
        if key in filters and not _is_empty_value(filters.get(key)):
            expr = _filter_expr(key)
            val = filters[key]
            if isinstance(val, list):
                clauses.append(f"{expr} = ANY(%s)")
            else:
                clauses.append(f"{expr} = %s")
            params.append(val)

    text_expr = (
        "LOWER(COALESCE(s.candidate_json->>'summary', '')) || ' ' || "
        "LOWER(COALESCE(NULLIF(p.action_text, ''), s.candidate_json->>'action_text', '')) || ' ' || "
        "LOWER(COALESCE(NULLIF(p.topic, ''), s.candidate_json->>'topic', '')) || ' ' || "
        "LOWER(COALESCE(NULLIF(p.source_quote, ''), s.candidate_json->>'source_quote', ''))"
    )
    kw_clauses = []
    for kw in keywords:
        kw_clauses.append(f"{text_expr} LIKE %s")
        params.append(f"%{kw}%")
    if kw_clauses:
        clauses.append("(" + " OR ".join(kw_clauses) + ")")

    where_sql = " AND ".join(clauses) if clauses else "TRUE"
    sql = (
        f"SELECT {select_clause}\n"
        f"{BASE_FROM_SQL.strip()}\n"
        f"WHERE {where_sql}\n"
        f"LIMIT {int(DEFAULT_LIMIT)}"
    )
    return sql, tuple(params)
