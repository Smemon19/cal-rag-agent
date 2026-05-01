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
INTENT_COMPLETE_LIST = "complete_list"
INTENT_APPROVAL_REQUIREMENT = "approval_requirement"
INTENT_EXPENSE_CATEGORY = "expense_category"
INTENT_TIMESHEET_ACTIVITY = "timesheet_activity"
INTENT_PROCEDURE_STEPS = "procedure_steps"
INTENT_DEFINITION = "definition"
INTENT_ELIGIBILITY = "eligibility"
INTENT_FALLBACK = "fallback"

INTENTS: frozenset[str] = frozenset(
    {
        INTENT_COMPLETE_LIST,
        INTENT_APPROVAL_REQUIREMENT,
        INTENT_EXPENSE_CATEGORY,
        INTENT_TIMESHEET_ACTIVITY,
        INTENT_PROCEDURE_STEPS,
        INTENT_DEFINITION,
        INTENT_ELIGIBILITY,
        INTENT_FALLBACK,
    }
)

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

Key columns:
- topic, subtopic       — THE primary filter columns. Always filter on these first.
- condition_text        — the "if/when" part of a policy rule
- action_text           — the "then/must do" part of a policy rule
- source_quote          — the verbatim text from the original document (always include this)
- summary               — a short plain-English summary of the row
- policy_category       — high-level bucket (e.g. payroll, billing, travel). Often NULL — only filter if it appears in the snapshot.
- applies_to, department, employee_type — who the policy applies to
- activity_type         — specific work type. Often NULL — only filter if it appears in the snapshot.
- bill_to, charge_code, is_billable — billing/charging info
- approval_required, approver — approval rules
- deadline_text, amount_threshold, threshold_unit — deadlines and limits
- exception_text, consequence_text — exceptions and consequences
- status                — always filter on 'active' unless told otherwise
"""


PLANNER_SYSTEM = f"""You are a search planner for an internal policy database. Your ONLY job is to output a JSON search spec. Do NOT answer the question. Do NOT write SQL.

{SCHEMA_CONTEXT}

=== HARD RULES — FOLLOW EXACTLY ===

RULE 1 — USE ONLY REAL TOPIC VALUES:
A DB Snapshot will be given to you showing every topic and subtopic currently in the database.
You MUST select topic and subtopic filter values ONLY from that exact list. Copy them verbatim, matching case exactly.
Never invent, paraphrase, or guess a topic that is not in the snapshot.
If multiple topics from the list are relevant, include all of them as an array — this is always better than picking just one.
If no topic perfectly matches, pick the closest one(s) from the list — do not leave the topic filter empty.

RULE 2 — OTHER FILTER VALUES:
Every other filter value (policy_category, activity_type, department, etc.) must also appear verbatim in the DB Snapshot.
If a value you want is not in the snapshot, omit that filter entirely.

RULE 3 — PREFER BROAD OVER NARROW:
Returning too many rows is better than returning zero rows.
When in doubt, include more topics in an array rather than picking just one.
If a question is about a broad area (e.g. timesheets, time tracking, PTO, holidays, expenses), always include
ALL related topics from the snapshot as an array — never bet on just one topic for broad questions.

RULE 4 — ALWAYS INCLUDE:
- "status": "active" in filters
- "source_quote" in requested_fields

RULE 5 — COMPARISON QUESTIONS:
If the question compares two things (A vs B), set "comparison": true and use arrays to capture both sides.

RULE 6 — SPECIFIC VS BROAD QUESTIONS:
- For SPECIFIC questions (a deadline, a yes/no rule, a single procedure): pick the single most relevant topic
  and add a subtopic filter if one from the snapshot clearly matches.
- For BROAD or COMPARISON questions (differences between two things, overview of a policy area): use an array.
- When unsure, prefer the topic with the MOST rows in the snapshot — a larger topic is more likely to contain
  the answer than a small single-row topic.

RULE 7 — TOPIC GUIDANCE (what each major topic covers):
Use these mappings to pick the right topic even when the question wording doesn't directly match the topic name:
- "Time Tracking": covers ALL timesheet rules — submission deadlines, who can fill out timesheets, daily entry
  requirements, billable vs overhead categories, social event charging restrictions, general timekeeping policy.
  IMPORTANT: questions about charging time at holiday parties, company events, tradeshows, or social gatherings
  belong here (subtopic "Social Events"), NOT under "Holidays" or "Marketing".
- "General Administration": covers internal admin duties AND rules about training other employees, onboarding,
  HR activities, IT, payroll, company equipment, team management.
- "Business Development": covers pre-proposal client meetings, go/no-go decisions, market research, cold calls.
- "Professional Development": covers approved training for the employee's own skills, continuing education.
  Note: if a question is about LEADING training for others, check "General Administration" — that is where
  the rule about training others lives, NOT Professional Development.
- "Marketing": covers unallowable promotional time, tradeshows, advertising, social event logistics.
- "Holidays": covers company holiday schedule, holiday time usage and carryover rules — NOT how to charge
  time at holiday parties (that is "Time Tracking").
- "Paid Time Off": covers PTO accrual, advance usage, request procedures, unscheduled absences.
- "Bid/Proposal": covers bid and proposal work, Ajera activity codes, RFQ/RFP preparation.
- "Billable Time": covers direct project labor charging rules.

=== OUTPUT FORMAT ===
Return JSON only — no markdown, no explanation outside JSON.
{{
  "table": "policies_v2",
  "filters": {{ "topic": "...", "status": "active" }},
  "requested_fields": ["topic", "subtopic", "condition_text", "action_text", "source_quote"],
  "comparison": false,
  "reasoning_note": "which topics you picked and why"
}}
"""


def classify_intent(question: str) -> str:
    """Classify the user's policy question with simple keyword rules."""
    q = f" {(question or '').strip().lower()} "
    compact = " ".join(q.split())

    if "approval" in compact or "require approval" in compact or "need approval" in compact:
        return INTENT_APPROVAL_REQUIREMENT

    if "timesheet" in compact and ("activity" in compact or "select" in compact):
        return INTENT_TIMESHEET_ACTIVITY

    if "category" in compact or "categorize" in compact:
        return INTENT_EXPENSE_CATEGORY

    expense_terms = ("client dinner", "dinner", "meal", "meals", "expense", "expenses", "travel")
    if "how do i charge" in compact and any(term in compact for term in expense_terms):
        return INTENT_EXPENSE_CATEGORY

    if "what are" in compact and (
        " all " in q
        or any(term in compact for term in ("holidays", "categories", "activity codes", "activities", "options"))
    ):
        return INTENT_COMPLETE_LIST

    if "how do i" in compact or "steps" in compact or "submit" in compact:
        return INTENT_PROCEDURE_STEPS

    if "what is" in compact or "define" in compact:
        return INTENT_DEFINITION

    if "can i" in compact or "allowed" in compact:
        return INTENT_ELIGIBILITY

    return INTENT_FALLBACK


def _append_requested_fields(search_spec: dict[str, Any], fields: tuple[str, ...]) -> None:
    requested_fields = list(search_spec.get("requested_fields") or [])
    for field in fields:
        if field in ALLOWED_REQUESTED_FIELDS and field not in requested_fields:
            requested_fields.append(field)
    search_spec["requested_fields"] = requested_fields


def _apply_intent_adjustments(search_spec: dict[str, Any], intent: str) -> dict[str, Any]:
    if intent == INTENT_COMPLETE_LIST:
        filters = dict(search_spec.get("filters") or {})
        kept_filters = {
            key: val
            for key, val in filters.items()
            if key in {"status", "topic", "batch_id", "document_id"}
        }
        search_spec["filters"] = kept_filters or {"status": filters.get("status", "active")}
        _append_requested_fields(search_spec, ("topic", "subtopic", "summary", "status", "source_quote"))
    elif intent in {INTENT_EXPENSE_CATEGORY, INTENT_TIMESHEET_ACTIVITY}:
        _append_requested_fields(
            search_spec,
            ("activity_type", "charge_code", "bill_to", "topic", "is_billable", "source_quote"),
        )
    return search_spec


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
        requested_fields = ["topic", "subtopic", "condition_text", "action_text", "source_quote"]

    # Always include the three core content fields — the formatter needs them to answer accurately.
    for required in ("condition_text", "action_text", "source_quote"):
        if required not in requested_fields:
            requested_fields.append(required)

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


def fetch_db_context() -> str:
    """
    Query the live database for distinct values in key filterable columns.
    Returns a formatted string to inject into the planner prompt so it
    filters on values that actually exist, not guessed ones.
    Returns an empty string if the DB is unreachable or empty.
    """
    try:
        from policy_engine.db import run_query

        count_rows = run_query("SELECT COUNT(*) AS n FROM policies_v2")
        total = int((count_rows[0].get("n") or 0)) if count_rows else 0

        if total == 0:
            return "The policies_v2 table is currently empty — no rows have been published yet."

        rows = run_query(
            """
            SELECT col, val FROM (
                SELECT 'topic' AS col, topic AS val FROM policies_v2 WHERE topic IS NOT NULL AND topic != ''
                UNION ALL
                SELECT 'subtopic', subtopic FROM policies_v2 WHERE subtopic IS NOT NULL AND subtopic != ''
                UNION ALL
                SELECT 'policy_category', policy_category FROM policies_v2 WHERE policy_category IS NOT NULL AND policy_category != ''
                UNION ALL
                SELECT 'department', department FROM policies_v2 WHERE department IS NOT NULL AND department != ''
                UNION ALL
                SELECT 'employee_type', employee_type FROM policies_v2 WHERE employee_type IS NOT NULL AND employee_type != ''
                UNION ALL
                SELECT 'activity_type', activity_type FROM policies_v2 WHERE activity_type IS NOT NULL AND activity_type != ''
                UNION ALL
                SELECT 'status', status FROM policies_v2 WHERE status IS NOT NULL AND status != ''
            ) t
            GROUP BY col, val
            ORDER BY col, val
            LIMIT 300
            """
        )

        by_col: dict[str, list[str]] = {}
        for r in rows:
            col = str(r.get("col") or "")
            val = str(r.get("val") or "").strip()
            if col and val:
                by_col.setdefault(col, []).append(val)

        lines = [f"Total rows in policies_v2: {total}", ""]
        lines.append("=== TOPICS (use ONLY these exact values for topic filters) ===")
        for val in sorted(by_col.get("topic", [])):
            lines.append(f"  - {val}")

        if by_col.get("subtopic"):
            lines.append("")
            lines.append("=== SUBTOPICS (use ONLY these exact values for subtopic filters) ===")
            for val in sorted(by_col.get("subtopic", [])):
                lines.append(f"  - {val}")

        for col in ("policy_category", "department", "employee_type", "activity_type", "status"):
            if by_col.get(col):
                lines.append("")
                lines.append(f"=== {col.upper()} values ===")
                for val in sorted(by_col[col]):
                    lines.append(f"  - {val}")

        return "\n".join(lines)
    except Exception:
        return ""


def plan_search(question: str, db_context: str = "") -> dict[str, Any]:
    """
    Call OpenAI (gpt-4o-mini) to produce a JSON search plan, then validate and normalize it.
    Raises ValueError on empty question or invalid planner output.
    Pass db_context (from fetch_db_context()) to ground the planner in real DB values.
    """
    q = (question or "").strip()
    if not q:
        raise ValueError("Question is empty.")

    if db_context:
        system_prompt = (
            PLANNER_SYSTEM
            + f"\n\n=== DB SNAPSHOT — YOU MUST ONLY USE VALUES FROM THIS LIST ===\n"
            f"{db_context}\n"
            f"=== END OF DB SNAPSHOT ==="
        )
        user_content = (
            f"REMINDER: You must pick topic/subtopic values ONLY from the DB Snapshot in the system prompt. "
            f"Do not invent values that are not in the list.\n\n"
            f"User question (plan retrieval only — do not answer it):\n{q}"
        )
    else:
        system_prompt = PLANNER_SYSTEM
        user_content = f"User question (plan retrieval only — do not answer it):\n{q}"

    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        temperature=0.0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    )

    content = response.choices[0].message.content
    if not content:
        raise ValueError("Planner returned no content.")

    parsed = _parse_planner_json(content)
    search_spec = validate_and_normalize_search_spec(parsed)
    detected_intent = classify_intent(q)
    print(f"[Intent] {detected_intent} for question: {q}")
    search_spec["intent"] = detected_intent
    return _apply_intent_adjustments(search_spec, detected_intent)
