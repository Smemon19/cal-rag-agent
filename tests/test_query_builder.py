from __future__ import annotations

from policy_engine.query_builder import COMPLETE_LIST_LIMIT, DEFAULT_LIMIT, build_query_from_spec


def _base_spec() -> dict:
    return {
        "table": "policies_v2",
        "filters": {"topic": "Holidays", "subtopic": "Calendar", "status": "active"},
        "requested_fields": ["topic", "subtopic", "condition_text", "action_text", "source_quote"],
        "comparison": False,
    }


def test_complete_list_uses_broader_query_behavior() -> None:
    spec = _base_spec()
    spec["intent"] = "complete_list"

    sql, params = build_query_from_spec(spec)

    assert f"LIMIT {COMPLETE_LIST_LIMIT}" in sql
    assert " OR " in sql
    assert "source_quote" in sql
    assert "SELECT DISTINCT COALESCE" in sql
    assert "SELECT DISTINCT p3.section_id::text" in sql
    assert "LOWER(COALESCE(NULLIF(p.topic" in sql
    assert "LOWER(COALESCE(NULLIF(p.subtopic" in sql
    assert params[0] == "active"
    assert "%holidays%" in params
    assert "%calendar%" in params


def test_non_complete_list_query_behavior_stays_unchanged() -> None:
    sql, params = build_query_from_spec(_base_spec())

    assert f"LIMIT {DEFAULT_LIMIT}" in sql
    assert "SELECT DISTINCT COALESCE" not in sql
    assert "SELECT DISTINCT p3.section_id::text" not in sql
    assert "LOWER(COALESCE(NULLIF(p.topic, ''), s.candidate_json->>'topic')) = LOWER(%s)" in sql
    assert "LOWER(COALESCE(NULLIF(p.subtopic, ''), s.candidate_json->>'subtopic')) = LOWER(%s)" in sql
    assert params == ("active", "Calendar", "Holidays")
