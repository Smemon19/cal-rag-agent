from __future__ import annotations

from policy_engine import planner


class _FakeMessage:
    def __init__(self, content: str):
        self.content = content


class _FakeChoice:
    def __init__(self, content: str):
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content: str):
        self.choices = [_FakeChoice(content)]


def _fake_openai_with_plan(plan_json: str):
    class _FakeCompletions:
        def create(self, **_kwargs):
            return _FakeResponse(plan_json)

    class _FakeChat:
        completions = _FakeCompletions()

    class _FakeOpenAI:
        chat = _FakeChat()

    return _FakeOpenAI


def test_classify_complete_list() -> None:
    assert planner.classify_intent("What are Raymond’s holidays?") == "complete_list"


def test_classify_approval_requirement() -> None:
    assert planner.classify_intent("Do I need approval for PD?") == "approval_requirement"


def test_classify_expense_category() -> None:
    assert planner.classify_intent("What category do I use for client dinner?") == "expense_category"
    assert planner.classify_intent("How do I charge a client dinner?") == "expense_category"


def test_classify_procedure_steps() -> None:
    assert planner.classify_intent("How do I submit an expense report?") == "procedure_steps"


def test_plan_search_attaches_intent_and_tunes_complete_list(monkeypatch) -> None:
    monkeypatch.setattr(
        planner,
        "OpenAI",
        _fake_openai_with_plan(
            """
            {
              "table": "policies_v2",
              "filters": {
                "topic": "Holidays",
                "subtopic": "Calendar",
                "activity_type": "Vacation",
                "status": "active"
              },
              "requested_fields": ["topic", "source_quote"],
              "comparison": false,
              "reasoning_note": "holiday list"
            }
            """
        ),
    )

    spec = planner.plan_search("What are Raymond's holidays?")

    assert spec["intent"] == "complete_list"
    assert spec["filters"] == {"topic": "Holidays", "status": "active"}
    assert "summary" in spec["requested_fields"]


def test_plan_search_prioritizes_charge_fields_for_expense_intent(monkeypatch) -> None:
    monkeypatch.setattr(
        planner,
        "OpenAI",
        _fake_openai_with_plan(
            """
            {
              "table": "policies_v2",
              "filters": {"topic": "Business Development", "status": "active"},
              "requested_fields": ["topic", "condition_text", "action_text", "source_quote"],
              "comparison": false,
              "reasoning_note": "client dinner"
            }
            """
        ),
    )

    spec = planner.plan_search("How do I charge a client dinner?")

    assert spec["intent"] == "expense_category"
    for field in ("activity_type", "charge_code", "bill_to", "topic"):
        assert field in spec["requested_fields"]
