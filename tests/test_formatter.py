from __future__ import annotations

import pytest

from policy_badger.engine import formatter


def _fake_generate_with_answer(answer: str):
    def fake_generate(system, user, **_kwargs):
        return answer

    return fake_generate


def _capturing_generate(captured: dict, answer: str):
    def fake_generate(system, user, **kwargs):
        captured.update(system=system, user=user, **kwargs)
        return answer

    return fake_generate


def test_zero_rows_returns_no_policy_found(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_generate(*_args, **_kwargs):
        raise AssertionError("The model should not be called when no rows are available")

    monkeypatch.setattr(formatter, "generate", fail_generate)

    answer = formatter.format_answer("What is the relocation policy?", [])

    assert answer == "No policy found in the retrieved data."


def test_rows_exist_no_direct_answer_returns_related_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        formatter,
        "generate",
        _fake_generate_with_answer("No policy found in the retrieved data"),
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
    monkeypatch.setattr(formatter, "generate", _fake_generate_with_answer(structured_answer))

    answer = formatter.format_answer(
        "Where should setup time go?",
        [{"topic": "Marketing", "action_text": "Charge setup time to Marketing."}],
    )

    assert answer == structured_answer


def test_unstructured_answer_is_wrapped_with_action_guidance(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        formatter,
        "generate",
        _fake_generate_with_answer("Yes, professional development time requires approval."),
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

    monkeypatch.setattr(formatter, "generate", _capturing_generate(captured, """Direct Answer
Use Marketing for event setup time.

Key Details
- Setup time belongs to Marketing.

What You Should Do
- Log setup time to Marketing."""))

    formatter.format_answer(
        "Where should setup time go?",
        [{"topic": "Marketing", "action_text": "Log setup time to Marketing."}],
    )

    user_prompt = captured["user"]
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

    monkeypatch.setattr(formatter, "generate", _capturing_generate(captured, "Direct Answer\nMock."))

    # Test expense_category
    formatter.format_answer("client dinner", [{"topic": "Meals"}], intent="expense_category")
    assert "emphasize the category FIRST" in captured["user"]

    # Test approval_requirement
    formatter.format_answer("approval", [{"topic": "Meals"}], intent="approval_requirement")
    assert "YES/NO first" in captured["user"]

    # Test complete_list
    formatter.format_answer("list", [{"topic": "Meals"}], intent="complete_list")
    assert "full list cleanly" in captured["user"]

    # Test definition
    formatter.format_answer("definition", [{"topic": "Meals"}], intent="definition")
    assert "clean definition first" in captured["user"]

    # Test procedure_steps
    formatter.format_answer("how to", [{"topic": "Meals"}], intent="procedure_steps")
    assert "step-by-step instructions" in captured["user"]


def test_unstructured_not_covered_reply_is_not_padded_with_unrelated_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        formatter,
        "generate",
        _fake_generate_with_answer("Direct Answer The provided policy records do not contain information about pets."),
    )
    rows = [{"topic": "PTO", "condition_text": "When an employee wants paid time off", "action_text": "Submit two weeks ahead."}]

    answer = formatter.format_answer("Can I bring my dog to work?", rows)

    assert "I found related policy records, but not a direct answer" in answer
    assert "Applies when" not in answer


def test_header_on_first_line_is_not_repeated(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        formatter,
        "generate",
        _fake_generate_with_answer("Direct Answer: Timesheets are due Friday by 5 PM."),
    )

    answer = formatter.format_answer("When are timesheets due?", [{"topic": "Timesheets", "action_text": "Submit by Friday 5pm."}])

    assert answer.count("Direct Answer") == 1
    assert "Timesheets are due Friday by 5 PM." in answer


def test_markdown_emphasis_is_removed(monkeypatch: pytest.MonkeyPatch) -> None:
    structured = """Direct Answer
**Yes**, approval is required.

Key Details
- **Approver:** your supervisor

What You Should Do
- **Obtain Preapproval:** ask before taking leave."""
    monkeypatch.setattr(formatter, "generate", _fake_generate_with_answer(structured))

    answer = formatter.format_answer("Do I need approval?", [{"topic": "Leave", "action_text": "Get approval."}])

    assert "**" not in answer
    assert "- Obtain Preapproval: ask before taking leave." in answer
    assert answer.startswith("Direct Answer\nYes, approval is required.")
