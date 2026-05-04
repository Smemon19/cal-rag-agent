from __future__ import annotations

from typing import Any

from policy_engine import service
from policy_engine.synonyms import expand_question


def test_client_dinner_expands_to_meal_and_expense_terms() -> None:
    expanded = expand_question("How do I charge a client dinner?")

    assert expanded.startswith("How do I charge a client dinner?")
    assert "meal expense client meeting" in expanded
    assert "expense category" in expanded


def test_pd_time_expands_to_professional_development() -> None:
    expanded = expand_question("Can I log PD time?")

    assert expanded.startswith("Can I log PD time?")
    assert "professional development" in expanded


def test_tqc_and_teqc_expand_to_same_term() -> None:
    tqc_expanded = expand_question("Where does TQC go?")
    teqc_expanded = expand_question("Where does TeQC go?")

    assert "total quality control" in tqc_expanded
    assert "total quality control" in teqc_expanded


def test_normal_question_without_synonym_is_unchanged() -> None:
    question = "When are timesheets due?"

    assert expand_question(question) == question


def test_service_passes_expanded_question_to_planner(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def fake_plan_search(question: str, db_context: str = "") -> dict[str, Any]:
        captured["question"] = question
        captured["db_context"] = db_context
        return {"fields": ["topic", "action_text"], "filters": {}}

    monkeypatch.setattr(service, "_get_db_context", lambda: "db context")
    monkeypatch.setattr(service, "plan_search", fake_plan_search)
    monkeypatch.setattr(service, "build_query_from_spec", lambda _spec: ("SELECT 1", ()))
    monkeypatch.setattr(service, "run_query", lambda _sql, _params=(): [{"topic": "Expenses"}])
    monkeypatch.setattr(service, "format_answer", lambda _question, _rows: "answer")

    result = service.answer_policy_question("How do I charge a client dinner?")

    assert result["source"] == "structured_policy_db"
    assert captured["question"].startswith("How do I charge a client dinner?")
    assert "meal expense client meeting" in captured["question"]
    assert "expense category" in captured["question"]
