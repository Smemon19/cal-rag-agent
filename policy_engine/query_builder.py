"""
Build safe, parameterized SELECT queries from a validated search spec.
Only allowlisted identifiers appear in SQL; values are always bound via %s / ANY(%s).
"""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

from policy_engine.planner import ALLOWED_FILTER_KEYS, ALLOWED_REQUESTED_FIELDS

DEFAULT_LIMIT = 10


def _quote_ident(name: str) -> str:
    """Quote identifier for PostgreSQL (allowlisted names only)."""
    if not name.replace("_", "").isalnum():
        raise ValueError(f"Invalid column name for SQL builder: {name!r}")
    return f'"{name}"'


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

    select_clause = ", ".join(_quote_ident(f) for f in fields)
    filters: dict[str, Any] = dict(search_spec.get("filters") or {})

    clauses: list[str] = []
    params: list[Any] = []

    for col in sorted(filters.keys()):
        if col not in ALLOWED_FILTER_KEYS:
            raise ValueError(f'Unknown filter column "{col}".')
        val = filters[col]
        ident = _quote_ident(col)

        if isinstance(val, list):
            if not val:
                continue
            clauses.append(f"{ident} = ANY(%s)")
            params.append(val)
        else:
            clauses.append(f"{ident} = %s")
            params.append(val)

    where_sql = " AND ".join(clauses) if clauses else "TRUE"
    sql = (
        f"SELECT {select_clause}\n"
        f"FROM policies_v2\n"
        f"WHERE {where_sql}\n"
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

    # Attempt 2: only policy_category + status (when current filters are broader than that)
    pc = filters.get("policy_category")
    st = filters.get("status", "active")
    minimal: dict[str, Any] = {"status": st}
    if pc is not None and not _is_empty_value(pc):
        minimal["policy_category"] = pc

    keys_set = set(filters.keys())
    if keys_set <= {"policy_category", "status"}:
        pass  # already minimal; no second variant
    elif minimal != filters:
        _add_variant(minimal)

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
