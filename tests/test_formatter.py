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

    assert "I found related policy records" in answer
    assert "could not find a direct policy" in answer
    assert "Time Tracking / Social Events" in answer
    assert "Do not include voluntary social event time" in answer
    assert "No policy found in the retrieved data" not in answer


def test_normal_successful_answer_is_preserved(monkeypatch: pytest.MonkeyPatch) -> None:
    direct_answer = "Charge setup time for voluntary social events to Marketing."
    monkeypatch.setattr(formatter, "OpenAI", _fake_openai_with_answer(direct_answer))

    answer = formatter.format_answer(
        "Where should setup time go?",
        [{"topic": "Marketing", "action_text": "Charge setup time to Marketing."}],
    )

    assert answer == direct_answer
