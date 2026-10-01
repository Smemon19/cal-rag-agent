from __future__ import annotations

import pytest

from policy_badger.engine import llm


class _FakeModels:
    def __init__(self, text):
        self.text = text
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return type("Response", (), {"text": self.text})()


def _install_fake_client(monkeypatch, text):
    models = _FakeModels(text)
    monkeypatch.setattr(llm, "_client", type("Client", (), {"models": models})())
    return models


def test_generate_sends_system_prompt_and_json_mode(monkeypatch):
    models = _install_fake_client(monkeypatch, ' {"table": "policies_v2"} ')
    monkeypatch.setenv("CHAT_LLM_MODEL", "gemini-test")

    out = llm.generate("SYS", "USER", temperature=0.0, json_output=True)

    assert out == '{"table": "policies_v2"}'
    call = models.calls[0]
    assert call["model"] == "gemini-test"
    assert call["contents"] == "USER"
    assert call["config"].system_instruction == "SYS"
    assert call["config"].response_mime_type == "application/json"
    assert call["config"].temperature == 0.0


def test_generate_plain_text_uses_default_model(monkeypatch):
    models = _install_fake_client(monkeypatch, "Direct Answer\nYes.")
    monkeypatch.delenv("CHAT_LLM_MODEL", raising=False)

    llm.generate("SYS", "USER", temperature=0.1)

    assert models.calls[0]["model"] == llm.DEFAULT_MODEL
    assert models.calls[0]["config"].response_mime_type is None


def test_generate_rejects_empty_response(monkeypatch):
    _install_fake_client(monkeypatch, None)

    with pytest.raises(ValueError):
        llm.generate("SYS", "USER", temperature=0.1)
