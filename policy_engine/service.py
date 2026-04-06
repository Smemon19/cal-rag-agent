"""
Orchestration: plan → validate (in planner) → SQL → Postgres → optional relaxed retry → formatted answer.
The LLM planner never executes SQL; all statements are built from allowlisted columns and parameters.
"""

from __future__ import annotations

import json
from typing import Any

from policy_engine.db import run_query
from policy_engine.formatter import format_answer
from policy_engine.planner import plan_search
from policy_engine.query_builder import build_query_from_spec, relaxed_specs_in_order


def _json_safe(obj: Any) -> Any:
    """Make result JSON-friendly (e.g. Decimal → str)."""
    return json.loads(json.dumps(obj, default=str))


def answer_policy_question(question: str) -> dict[str, Any]:
    """
    End-to-end policy Q&A against policies_v2.

    Flow:
    1. LLM search planner → validated search spec (JSON only, no answers).
    2. Build parameterized SELECT (LIMIT 10).
    3. Run query.
    4. If zero rows, try relaxed specs in order (drop activity_type, then category+status only).
    5. Formatter LLM produces the user-facing answer from rows only.
    """
    q = (question or "").strip()
    if not q:
        raise ValueError("Question is empty.")

    # Step 1–2: planner validates/normalizes the spec internally.
    search_spec = plan_search(q)

    # Step 3: primary SQL
    sql, params = build_query_from_spec(search_spec)
    rows = run_query(sql, params)

    used_relaxed = False
    relaxed_sql: str | None = None
    relaxed_params: list[Any] = []

    # Step 4: broader retries (still parameterized SELECT, same allowlists)
    if not rows:
        for spec_r in relaxed_specs_in_order(search_spec):
            sql_r, params_r = build_query_from_spec(spec_r)
            rows_try = run_query(sql_r, params_r)
            if rows_try:
                rows = rows_try
                used_relaxed = True
                relaxed_sql = sql_r
                relaxed_params = list(params_r)
                break

    # Step 5: answer from rows only
    answer = format_answer(q, rows)

    return {
        "question": q,
        "search_spec": search_spec,
        "sql": sql,
        "params": list(params),
        "relaxed_sql": relaxed_sql,
        "relaxed_params": relaxed_params,
        "rows": _json_safe(rows),
        "row_count": len(rows),
        "answer": answer,
        "used_relaxed_query": used_relaxed,
        "source": "structured_policy_db",
    }
