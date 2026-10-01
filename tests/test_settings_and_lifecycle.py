"""
Tests for the settings, conversation, policy-lifecycle and session-revalidation
work. Every database call is monkeypatched, so these never touch a real server.
"""

import pytest
from fastapi import HTTPException

from policy_badger.engine import auth_db, conversations_db, policy_admin_db, settings_db


# ── Settings sanitizing ──────────────────────────────────────────────────────

def test_sanitize_rejects_unknown_keys_and_clamps_ranges():
    dirty = {
        "length": 99,
        "precision": -5,
        "preset": "not-a-preset",
        "format": "exotic",
        "reading": "leet",
        "styles": ["Concise", "Definitely Not A Style"],
        "evil": "drop table",
    }
    clean = settings_db.sanitize_settings(dirty)

    assert clean["length"] == 2                 # clamped to the top of the range
    assert clean["precision"] == 0              # clamped to the bottom
    assert clean["preset"] == "balanced"        # invalid value falls back
    assert clean["format"] == "structured"
    assert clean["reading"] == "plain"
    assert clean["styles"] == ["Concise"]       # unknown style dropped
    assert "evil" not in clean                  # arbitrary keys never survive


def test_sanitize_caps_prompt_count_and_length():
    clean = settings_db.sanitize_settings({"prompts": ["q" * 500] + ["x"] * 20})
    assert len(clean["prompts"]) <= settings_db.MAX_PROMPTS
    assert all(len(p) <= settings_db.MAX_PROMPT_LEN for p in clean["prompts"])


def test_sanitize_drops_blank_prompts():
    clean = settings_db.sanitize_settings({"prompts": ["  ", "", "real question"]})
    assert clean["prompts"] == ["real question"]


def test_non_dict_input_returns_defaults():
    assert settings_db.sanitize_settings("nonsense") == settings_db.DEFAULT_SETTINGS
    assert settings_db.sanitize_settings(None) == settings_db.DEFAULT_SETTINGS


def test_directives_empty_when_nothing_to_say():
    assert settings_db.build_answer_directives({}) == ""


def test_directives_reflect_choices():
    text = settings_db.build_answer_directives({
        "length": 0,
        "format": "bullets",
        "preset": "strict",
        "toggles": {"sources": True, "honest": False, "counts": False},
        "styles": ["Name the approver"],
    })
    assert "LENGTH" in text and "short" in text.lower()
    assert "bullet" in text.lower()
    assert "PRECISION" in text
    assert "Cite the originating policy" in text
    assert "approving role" in text
    # A toggle that is off must not appear.
    assert "how many were found" not in text


# ── Session revalidation ─────────────────────────────────────────────────────

class FakeRequest:
    def __init__(self, session):
        self.session = session


def _session(**extra):
    base = {
        "user_id": "u1",
        "username": "someone",
        "role": "employee",
        "company_id": "c1",
    }
    base.update(extra)
    return base


def test_deactivated_user_is_signed_out_mid_session(monkeypatch):
    monkeypatch.setattr(
        auth_db, "run_query",
        lambda sql, params=None: [{"role": "employee", "is_active": False, "company_id": "c1"}],
    )
    request = FakeRequest(_session())

    with pytest.raises(HTTPException) as excinfo:
        auth_db.require_login(request)

    assert excinfo.value.status_code == 401
    assert request.session == {}      # the session is cleared, not just rejected


def test_deleted_user_is_signed_out(monkeypatch):
    monkeypatch.setattr(auth_db, "run_query", lambda sql, params=None: [])
    request = FakeRequest(_session())

    with pytest.raises(HTTPException) as excinfo:
        auth_db.require_login(request)

    assert excinfo.value.status_code == 401
    assert request.session == {}


def test_demotion_takes_effect_without_relogin(monkeypatch):
    """A user demoted from manager_admin loses admin routes on the next check."""
    monkeypatch.setattr(
        auth_db, "run_query",
        lambda sql, params=None: [{"role": "employee", "is_active": True, "company_id": "c1"}],
    )
    request = FakeRequest(_session(role="manager_admin"))

    user = auth_db.require_login(request)
    assert user["role"] == "employee"
    assert request.session["role"] == "employee"

    with pytest.raises(HTTPException) as excinfo:
        auth_db.require_roles(request, {"manager_admin", "super_admin"})
    assert excinfo.value.status_code == 403


def test_revalidation_is_cached_for_a_window(monkeypatch):
    calls = []

    def counting_query(sql, params=None):
        calls.append(sql)
        return [{"role": "employee", "is_active": True, "company_id": "c1"}]

    monkeypatch.setattr(auth_db, "run_query", counting_query)
    request = FakeRequest(_session())

    for _ in range(5):
        auth_db.require_login(request)

    assert len(calls) == 1, "revalidation should be cached, not run per request"


def test_database_outage_does_not_sign_everyone_out(monkeypatch):
    def boom(sql, params=None):
        raise RuntimeError("connection refused")

    monkeypatch.setattr(auth_db, "run_query", boom)
    request = FakeRequest(_session(role="manager_admin"))

    user = auth_db.require_login(request)
    assert user["role"] == "manager_admin"   # falls back to the cached claims
    assert request.session["user_id"] == "u1"


# ── Policy lifecycle ─────────────────────────────────────────────────────────

def test_quote_ident_rejects_injection():
    with pytest.raises(ValueError):
        policy_admin_db._quote_ident('topic"; DROP TABLE policies_v2; --')
    assert policy_admin_db._quote_ident("topic") == '"topic"'


def test_update_rejects_non_editable_columns(monkeypatch):
    monkeypatch.setattr(policy_admin_db, "_table_columns", lambda: {"policy_id", "topic", "status"})

    with pytest.raises(HTTPException) as excinfo:
        policy_admin_db.update_policy("actor", "p1", {"policy_id": "hijacked", "status": "active"})

    assert excinfo.value.status_code == 400   # nothing editable was supplied


def test_policy_cannot_supersede_itself(monkeypatch):
    monkeypatch.setattr(
        policy_admin_db, "_table_columns",
        lambda: {"policy_id", "status", "superseded_by_policy_id"},
    )
    with pytest.raises(HTTPException) as excinfo:
        policy_admin_db.supersede_policy("actor", "p1", "p1")
    assert excinfo.value.status_code == 400


def test_archive_requires_the_migration(monkeypatch):
    monkeypatch.setattr(policy_admin_db, "_table_columns", lambda: {"policy_id", "topic"})
    with pytest.raises(HTTPException) as excinfo:
        policy_admin_db.archive_policy("actor", "p1")
    assert excinfo.value.status_code == 503   # migration 0005 has not been applied


def test_list_rejects_unknown_status(monkeypatch):
    monkeypatch.setattr(policy_admin_db, "_table_columns", lambda: {"policy_id", "status"})
    with pytest.raises(HTTPException) as excinfo:
        policy_admin_db.list_policies(status="whatever")
    assert excinfo.value.status_code == 400


# ── Conversations ────────────────────────────────────────────────────────────

def test_title_derived_from_question():
    assert conversations_db._derive_title("  How do I   file expenses? ") == "How do I file expenses?"
    assert conversations_db._derive_title("") == "New conversation"

    long_title = conversations_db._derive_title("word " * 100)
    assert len(long_title) <= conversations_db.MAX_TITLE_LEN


def test_other_users_conversation_is_not_found(monkeypatch):
    # The ownership check filters by user_id, so a foreign id returns no rows.
    monkeypatch.setattr(conversations_db, "run_query", lambda sql, params=None: [])

    with pytest.raises(HTTPException) as excinfo:
        conversations_db.get_conversation("user-a", "conversation-owned-by-b")
    assert excinfo.value.status_code == 404

    with pytest.raises(HTTPException) as excinfo:
        conversations_db.delete_conversation("user-a", "conversation-owned-by-b")
    assert excinfo.value.status_code == 404


def test_rename_rejects_blank_title(monkeypatch):
    monkeypatch.setattr(conversations_db, "run_query", lambda sql, params=None: [{"1": 1}])
    with pytest.raises(HTTPException) as excinfo:
        conversations_db.rename_conversation("user-a", "c1", "   ")
    assert excinfo.value.status_code == 400


def test_malformed_conversation_id_is_not_found_without_querying(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("database should not be queried for a malformed id")

    monkeypatch.setattr(conversations_db, "run_query", fail)
    for call in (
        lambda: conversations_db.get_conversation("u1", "None"),
        lambda: conversations_db.delete_conversation("u1", "not-a-uuid"),
    ):
        with pytest.raises(HTTPException) as exc:
            call()
        assert exc.value.status_code == 404
