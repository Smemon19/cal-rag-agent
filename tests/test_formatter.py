from __future__ import annotations

import pytest

from policy_engine import formatter


class _FakeMessage:
    def __init__(self, content: str):
        self.content = content


class _FakeChoice:
    def __init__(self, content: str):
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content: str):
        self.choices = [_FakeChoice(content)]


def _fake_openai_with_answer(answer: str):
    class _FakeCompletions:
        def create(self, **_kwargs):
            return _FakeResponse(answer)

    class _FakeChat:
        completions = _FakeCompletions()

    class _FakeOpenAI:
        chat = _FakeChat()

    return _FakeOpenAI


def test_zero_rows_returns_no_policy_found(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_openai():
        raise AssertionError("OpenAI should not be called when no rows are available")

    monkeypatch.setattr(formatter, "OpenAI", fail_openai)

    answer = formatter.format_answer("What is the relocation policy?", [])

    assert answer == "No policy found in the retrieved data."


def test_rows_exist_no_direct_answer_returns_related_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        formatter,
        "OpenAI",
        _fake_openai_with_answer("No policy found in the retrieved data"),
    )
    rows = [
        {
            "topic": "Time Tracking",
            "subtopic": "Social Events",
            "action_text": "Do not include voluntary social event time on timesheets unless helping with setup.",
            "source_quote": "Participation in social events is voluntary.",
        }
    ]

    answer = formatter.format_answer("Can I bill a company picnic?", rows)

    assert "Direct Answer" in answer
    assert "Key Details" in answer
    assert "What You Should Do" in answer
    assert "I found related policy records" in answer
    assert "not a direct answer" in answer
    assert "Time Tracking / Social Events" in answer
    assert "Do not include voluntary social event time" in answer
    assert "No policy found in the retrieved data" not in answer


def test_structured_successful_answer_is_preserved(monkeypatch: pytest.MonkeyPatch) -> None:
    structured_answer = """Direct Answer
Charge setup time for voluntary social events to Marketing.

Key Details
- Setup and breakdown time should be entered to Marketing.

What You Should Do
- Log only setup or breakdown time to Marketing."""
    monkeypatch.setattr(formatter, "OpenAI", _fake_openai_with_answer(structured_answer))

    answer = formatter.format_answer(
        "Where should setup time go?",
        [{"topic": "Marketing", "action_text": "Charge setup time to Marketing."}],
    )

    assert answer == structured_answer


def test_unstructured_answer_is_wrapped_with_action_guidance(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        formatter,
        "OpenAI",
        _fake_openai_with_answer("Yes, professional development time requires approval."),
    )
    rows = [
        {
            "topic": "Professional Development",
            "approval_required": True,
            "approver": "Manager",
            "condition_text": "Professional development time must have specific learning goals.",
            "action_text": "Get approval before logging professional development time.",
            "exception_text": "The 120-hour limit is a rare maximum and approval is not guaranteed.",
        }
    ]

    answer = formatter.format_answer("Does PD time need approval?", rows)

    assert "Direct Answer" in answer
    assert "Key Details" in answer
    assert "Important Notes / Exceptions" in answer
    assert "What You Should Do" in answer
    assert "Approval required: Yes. Approver: Manager." in answer
    assert "Professional development time must have specific learning goals." in answer
    assert "The 120-hour limit is a rare maximum and approval is not guaranteed." in answer
    assert "Get approval before logging professional development time." in answer


def test_formatter_prompt_prioritizes_policy_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class _FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return _FakeResponse(
                """Direct Answer
Use Marketing for event setup time.

Key Details
- Setup time belongs to Marketing.

What You Should Do
- Log setup time to Marketing."""
            )

    class _FakeChat:
        completions = _FakeCompletions()

    class _FakeOpenAI:
        chat = _FakeChat()

    monkeypatch.setattr(formatter, "OpenAI", _FakeOpenAI)

    formatter.format_answer(
        "Where should setup time go?",
        [{"topic": "Marketing", "action_text": "Log setup time to Marketing."}],
    )

    messages = captured["messages"]
    user_prompt = messages[1]["content"]
    assert "Direct Answer" in user_prompt
    assert "Key Details" in user_prompt
    assert "Important Notes / Exceptions" in user_prompt
    assert "What You Should Do" in user_prompt
    assert "approval_required" in user_prompt
    assert "exception_text" in user_prompt
    assert "condition_text" in user_prompt
    assert "action_text" in user_prompt


def test_formatter_adds_intent_instructions(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class _FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return _FakeResponse("Direct Answer\nMock.")

    class _FakeChat:
        completions = _FakeCompletions()

    class _FakeOpenAI:
        chat = _FakeChat()

    monkeypatch.setattr(formatter, "OpenAI", _FakeOpenAI)

    # Test expense_category
    formatter.format_answer("client dinner", [{"topic": "Meals"}], intent="expense_category")
    assert "emphasize the category FIRST" in captured["messages"][1]["content"]

    # Test approval_requirement
    formatter.format_answer("approval", [{"topic": "Meals"}], intent="approval_requirement")
    assert "YES/NO first" in captured["messages"][1]["content"]

    # Test complete_list
    formatter.format_answer("list", [{"topic": "Meals"}], intent="complete_list")
    assert "full list cleanly" in captured["messages"][1]["content"]

    # Test definition
    formatter.format_answer("definition", [{"topic": "Meals"}], intent="definition")
    assert "clean definition first" in captured["messages"][1]["content"]

    # Test procedure_steps
    formatter.format_answer("how to", [{"topic": "Meals"}], intent="procedure_steps")
    assert "step-by-step instructions" in captured["messages"][1]["content"]
