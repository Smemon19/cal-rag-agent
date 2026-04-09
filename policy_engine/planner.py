"""
LLM-based search planner: question → JSON search spec for policies_v2.
Planner outputs structured filters only; it must not answer the question or execute SQL.
"""

from __future__ import annotations

import json
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)

ALLOWED_TABLE = "policies_v2"

ALLOWED_FILTER_KEYS: frozenset[str] = frozenset(
    {
        "policy_category",
        "topic",
        "subtopic",
        "document_id",
        "batch_id",
        "applies_to",
        "department",
        "employee_type",
        "activity_type",
        "bill_to",
        "charge_code",
        "approval_required",
        "approver",
        "status",
        "receipt_required",
        "annual_cap",
        "region_code",
        "approval_frequency",
    }
)

ALLOWED_REQUESTED_FIELDS: frozenset[str] = frozenset(
    {
        "document_id",
        "batch_id",
        "summary",
        "policy_category",
        "topic",
        "subtopic",
        "applies_to",
        "department",
        "employee_type",
        "activity_type",
        "condition_text",
        "action_text",
        "bill_to",
        "charge_code",
        "is_billable",
        "approval_required",
        "approver",
        "deadline_text",
        "amount_threshold",
        "threshold_unit",
        "exception_text",
        "consequence_text",
        "source_quote",
        "status",
        "receipt_required",
        "annual_cap",
        "region_code",
        "approval_frequency",
        "policy_precedence",
        "override_policy_id",
    }
)

SCHEMA_CONTEXT = """
Table: policies_v2

Columns (for planning retrieved fields and filters):
- policy_id, section_id  — row identifiers (do not request in requested_fields for this app; use other columns)
- document_id, batch_id  — optional retrieval scope for a specific ingestion document/batch
- policy_category   — high-level bucket (e.g. payroll, billing, marketing, travel, overtime)
- summary           — normalized semantic summary (from adaptive staging when available)
- topic, subtopic
- applies_to, department, employee_type
- activity_type     — more specific work type (e.g. internal marketing, client campaign, overtime)
- condition_text
- action_text
- bill_to           — where time/cost should be charged
- charge_code       — billing or overhead code
- is_billable
- approval_required, approver
- deadline_text
- amount_threshold, threshold_unit
- exception_text, consequence_text
- source_quote
- status            — should usually be 'active' for current policy

Meanings for the planner:
- policy_category is the high-level bucket.
- activity_type is the more specific activity/work type.
- bill_to is the charging destination; charge_code is the code to use.
- Prefer status = 'active' unless the user clearly asks for inactive/archived policy.
"""


PLANNER_SYSTEM = f"""You are a database search planner for an internal policy system.

{SCHEMA_CONTEXT}

Your job:
- Read the user's question and propose a structured JSON search specification for table {ALLOWED_TABLE}.
- Do NOT answer the question or explain policy outcomes.
- Do NOT write SQL.
- Use only the schema above. If unsure about a filter value, omit that filter rather than guessing.
- Prefer broader but relevant filters over overly narrow ones that may return zero rows.
- For comparison questions (e.g. A vs B), set "comparison": true and use arrays in filters when multiple values apply
  (e.g. "activity_type": ["internal marketing", "client campaign"] or multiple categories if needed).
- Always include "status": "active" in filters unless there is a strong reason not to.
- Include enough requested_fields to support the user's question: billing/charge/approvals/exceptions/deadlines as relevant.
- Return JSON only (no markdown, no prose outside JSON).

Required JSON shape:
{{
  "table": "policies_v2",
  "filters": {{ "...": "scalar or array of strings/booleans as appropriate" }},
  "requested_fields": ["column names to SELECT"],
  "comparison": false,
  "reasoning_note": "brief note on filter choices (not an answer to the user)"
}}
"""


def _strip_json_fences(text: str) -> str:
    """Remove optional ``` / ```json markdown fences from model output."""
    t = (text or "").strip()
    if not t.startswith("```"):
        return t
    lines = t.split("\n")
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _parse_planner_json(content: str) -> dict[str, Any]:
    stripped = _strip_json_fences(content)
    if not stripped:
        raise ValueError("Planner returned empty content.")
    try:
        return json.loads(stripped)
    except json.JSONDecodeError as e:
        raise ValueError(f"Planner returned invalid JSON: {e}") from e


def _normalize_filter_value(val: Any) -> str | bool | list[str | bool]:
    """Normalize a filter value to scalar str/bool or list of str/bool; drop empty list entries."""
    if val is None:
        raise ValueError("Filter values must not be null; omit the key instead.")

    if isinstance(val, bool):
        return val

    if isinstance(val, (int, float)):
        return str(val)

    if isinstance(val, str):
        s = val.strip()
        if not s:
            raise ValueError("Empty string filter values are not allowed; omit the key.")
        return s

    if isinstance(val, list):
        out: list[str | bool] = []
        for item in val:
            if item is None:
                continue
            if isinstance(item, bool):
                out.append(item)
            elif isinstance(item, (int, float)):
                out.append(str(item))
            else:
                si = str(item).strip()
                if si:
                    out.append(si)
        if not out:
            raise ValueError("Filter list became empty after normalization; omit the key.")
        if len(out) == 1:
            return out[0]
        return out

    raise ValueError(f"Unsupported filter value type: {type(val).__name__}")


def validate_and_normalize_search_spec(raw: dict[str, Any]) -> dict[str, Any]:
    """
    Validate planner JSON against allowlists; normalize filters and requested_fields.
    - Rejects unknown table, filter keys, or requested fields.
    - Ensures requested_fields is non-empty and includes source_quote for evidence.
    - Adds status = active if status not present in filters.
    - Preserves comparison boolean (default False).
    """
    if not isinstance(raw, dict):
        raise ValueError("Search spec must be a JSON object.")

    table = raw.get("table")
    if table != ALLOWED_TABLE:
        raise ValueError(f'Invalid table "{table}". Only "{ALLOWED_TABLE}" is allowed.')

    filters_in = raw.get("filters")
    if not isinstance(filters_in, dict):
        raise ValueError('"filters" must be an object.')

    filters: dict[str, str | bool | list[str | bool]] = {}
    for key, val in filters_in.items():
        if key not in ALLOWED_FILTER_KEYS:
            raise ValueError(f'Unknown filter key "{key}". Not in allowlist.')
        try:
            filters[key] = _normalize_filter_value(val)
        except ValueError as e:
            raise ValueError(f'Invalid filter value for "{key}": {e}') from e

    if "status" not in filters:
        filters["status"] = "active"

    rf = raw.get("requested_fields")
    if not isinstance(rf, list) or not rf:
        rf = ["topic", "subtopic", "summary", "condition_text", "action_text", "source_quote"]

    requested_fields: list[str] = []
    for col in rf:
        if not isinstance(col, str) or not col.strip():
            raise ValueError("requested_fields entries must be non-empty strings.")
        c = col.strip()
        if c not in ALLOWED_REQUESTED_FIELDS:
            continue
        if c not in requested_fields:
            requested_fields.append(c)

    if not requested_fields:
        requested_fields = ["topic", "subtopic", "summary", "condition_text", "action_text", "source_quote"]

    if "source_quote" not in requested_fields:
        requested_fields.append("source_quote")

    comparison = raw.get("comparison", False)
    if not isinstance(comparison, bool):
        raise ValueError('"comparison" must be a boolean.')

    reasoning = raw.get("reasoning_note", "")
    if reasoning is not None and not isinstance(reasoning, str):
        reasoning = str(reasoning)

    return {
        "table": ALLOWED_TABLE,
        "filters": filters,
        "requested_fields": requested_fields,
        "comparison": comparison,
        "reasoning_note": reasoning or "",
    }


def plan_search(question: str) -> dict[str, Any]:
    """
    Call OpenAI (gpt-4o-mini) to produce a JSON search plan, then validate and normalize it.
    Raises ValueError on empty question or invalid planner output.
    """
    q = (question or "").strip()
    if not q:
        raise ValueError("Question is empty.")

    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        temperature=0.1,
        messages=[
            {"role": "system", "content": PLANNER_SYSTEM},
            {
                "role": "user",
                "content": f"User question (plan retrieval only; do not answer):\n{q}",
            },
        ],
    )

    content = response.choices[0].message.content
    if not content:
        raise ValueError("Planner returned no content.")

    parsed = _parse_planner_json(content)
    return validate_and_normalize_search_spec(parsed)
