"""
Tests for telling newly extracted policies apart from already-published ones.

These decide what an admin sees as "new" versus "already in the database", so
the thresholds and the tie-breaking are worth pinning down.
"""

import pytest
from fastapi import HTTPException

from policy_badger.ingestion import dedup, document_upload


def _existing(policy_id, topic, action, condition="", subtopic=""):
    row = {
        "policy_id": policy_id,
        "topic": topic,
        "subtopic": subtopic,
        "condition_text": condition,
        "action_text": action,
        "source_quote": "",
    }
    row["_tokens"] = dedup._candidate_tokens(row)
    row["_topic"] = topic.strip().lower()
    row["_subtopic"] = subtopic.strip().lower()
    return row


PUBLISHED = [
    _existing(
        1, "Expenses",
        "Submit expense reports within thirty days of incurring the expense, attaching itemised receipts.",
        subtopic="Reimbursement",
    ),
    _existing(
        2, "Timesheets",
        "Timesheets must be submitted by noon on the Monday following the end of each pay period.",
    ),
    _existing(
        3, "Holidays",
        "Employees receive two floating holidays which may be taken at any time with prior supervisor approval.",
    ),
]


# ── Similarity ───────────────────────────────────────────────────────────────

def test_similarity_is_symmetric_and_bounded():
    a = dedup._tokens("submit expense reports within thirty days")
    b = dedup._tokens("expense reports must be submitted within thirty days")
    assert dedup.similarity(a, b) == dedup.similarity(b, a)
    assert 0.0 <= dedup.similarity(a, b) <= 1.0


def test_empty_text_scores_zero():
    assert dedup.similarity(set(), dedup._tokens("anything")) == 0.0
    assert dedup.similarity(dedup._tokens("anything"), set()) == 0.0


def test_stopwords_do_not_create_false_matches():
    a = dedup._tokens("the employee must be in the office")
    b = dedup._tokens("the policy should be of their own")
    # Both are almost entirely stopwords; nothing substantive overlaps.
    assert dedup.similarity(a, b) == 0.0


def test_overlap_beats_jaccard_for_verbose_restatement():
    """A wordier restatement of the same rule must still read as a duplicate."""
    short = dedup._tokens("Submit expense reports within thirty days with receipts.")
    verbose = dedup._tokens(
        "Employees are required to submit their expense reports within thirty days "
        "of incurring the expense, and must attach itemised receipts covering every "
        "line item claimed, otherwise reimbursement may be delayed."
    )
    assert dedup.similarity(short, verbose) >= dedup.DUPLICATE_THRESHOLD


# ── Classification ───────────────────────────────────────────────────────────

def test_restated_policy_is_a_duplicate():
    candidate = {
        "topic": "Expenses",
        "subtopic": "Reimbursement",
        "action_text": "Submit expense reports within thirty days, attaching itemised receipts.",
        "publishable": True,
    }
    result = dedup.classify_candidate(candidate, PUBLISHED)
    assert result["verdict"] == "duplicate"
    assert result["matched_policy_id"] == "1"


def test_genuinely_new_policy_is_new():
    candidate = {
        "topic": "Remote Work",
        "action_text": "Employees working remotely must maintain a secure broadband connection "
                       "and be reachable during core collaboration hours.",
        "publishable": True,
    }
    result = dedup.classify_candidate(candidate, PUBLISHED)
    assert result["verdict"] == "new"
    assert result["matched_policy_id"] is None


def test_moderate_rewording_of_existing_policy_is_flagged_as_update():
    """Same rule, changed details: the reviewer should see it as a revision."""
    candidate = {
        "topic": "Expenses",
        "action_text": "Submit expense reports within fourteen days of incurring the expense; "
                       "receipts are required for claims over fifty dollars.",
        "publishable": True,
    }
    result = dedup.classify_candidate(candidate, PUBLISHED)
    assert result["verdict"] == "update"
    assert result["matched_policy_id"] == "1"


def test_heavily_reworded_revision_surfaces_as_new():
    """
    A rewrite that shares almost no wording is reported as new, not as an
    update. Only 'timesheets' and 'submitted' survive in common here, and any
    threshold low enough to catch it would mislabel unrelated policies that
    merely share a topic. Surfacing it as new is the safe failure: the admin
    ingests it and supersedes the old row from the Policy Library, whereas a
    wrong 'update' label invites silently replacing the wrong policy.
    """
    candidate = {
        "topic": "Timesheets",
        "action_text": "Timesheets must be submitted by close of business Friday, "
                       "reviewed by a supervisor before payroll processes them.",
        "publishable": True,
    }
    result = dedup.classify_candidate(candidate, PUBLISHED)
    assert result["verdict"] == "new"


def test_same_wording_under_a_different_topic_is_not_an_update():
    """Topic agreement is required for 'update'; without it, it stays 'new'."""
    candidate = {
        "topic": "Contractors",
        "action_text": "Timesheets must be submitted by close of business Friday, "
                       "reviewed by a supervisor before payroll runs.",
        "publishable": True,
    }
    result = dedup.classify_candidate(candidate, PUBLISHED)
    assert result["verdict"] in ("new", "duplicate")
    if result["verdict"] == "new":
        assert result["matched_policy_id"] is None


def test_candidate_with_no_text_is_new_not_duplicate():
    result = dedup.classify_candidate({"topic": "Expenses", "publishable": True}, PUBLISHED)
    assert result["verdict"] == "new"


def test_classification_against_empty_database_is_always_new():
    candidate = {"topic": "Expenses", "action_text": "Submit reports promptly.", "publishable": True}
    assert dedup.classify_candidate(candidate, [])["verdict"] == "new"


# ── Batch review ─────────────────────────────────────────────────────────────

def test_review_batch_counts_and_orders(monkeypatch):
    monkeypatch.setattr(dedup, "load_existing_policies", lambda: PUBLISHED)

    staged = [
        {"extraction_id": "e1", "confidence": 0.9, "candidate_json": {
            "topic": "Expenses", "subtopic": "Reimbursement", "publishable": True,
            "action_text": "Submit expense reports within thirty days, attaching itemised receipts."}},
        {"extraction_id": "e2", "confidence": 0.8, "candidate_json": {
            "topic": "Parental Leave", "publishable": True,
            "action_text": "Primary caregivers receive twelve weeks of paid parental leave."}},
        {"extraction_id": "e3", "confidence": 0.2, "candidate_json": {
            "topic": None, "publishable": False, "action_text": ""}},
    ]

    review = dedup.review_batch(staged)

    assert review["total"] == 3
    assert review["counts"]["duplicate"] == 1
    assert review["counts"]["new"] == 1
    assert review["counts"]["unpublishable"] == 1
    # New first, noise last, so the reviewer reads top-down.
    assert review["items"][0]["verdict"] == "new"
    assert review["items"][-1]["verdict"] == "unpublishable"


def test_review_batch_accepts_json_encoded_candidates(monkeypatch):
    """candidate_json comes back as a string from some drivers."""
    monkeypatch.setattr(dedup, "load_existing_policies", lambda: PUBLISHED)
    staged = [{
        "extraction_id": "e1",
        "confidence": 0.9,
        "candidate_json": '{"topic": "Expenses", "publishable": true, '
                          '"action_text": "Submit expense reports within thirty days with itemised receipts."}',
    }]
    review = dedup.review_batch(staged)
    assert review["items"][0]["verdict"] == "duplicate"


def test_malformed_candidate_json_does_not_crash_review(monkeypatch):
    monkeypatch.setattr(dedup, "load_existing_policies", lambda: PUBLISHED)
    staged = [{"extraction_id": "e1", "confidence": 0.1, "candidate_json": "not json at all"}]
    review = dedup.review_batch(staged)
    assert review["total"] == 1


# ── Upload guards ────────────────────────────────────────────────────────────

def test_unsupported_file_type_is_rejected():
    with pytest.raises(HTTPException) as excinfo:
        document_upload._check_upload("payroll.xlsx", 1000)
    assert excinfo.value.status_code == 400


def test_empty_file_is_rejected():
    with pytest.raises(HTTPException) as excinfo:
        document_upload._check_upload("handbook.pdf", 0)
    assert excinfo.value.status_code == 400


def test_oversized_file_is_rejected():
    with pytest.raises(HTTPException) as excinfo:
        document_upload._check_upload("handbook.pdf", document_upload.MAX_UPLOAD_BYTES + 1)
    assert excinfo.value.status_code == 413


def test_supported_types_pass():
    for name in ("handbook.pdf", "policy.DOCX", "notes.txt", "readme.md"):
        assert document_upload._check_upload(name, 2048).startswith(".")


def test_publishing_nothing_is_rejected():
    with pytest.raises(HTTPException) as excinfo:
        document_upload.publish_selected(batch_id="bat_1", extraction_ids=[], published_by="u1")
    assert excinfo.value.status_code == 400


def test_cannot_publish_extractions_from_another_batch(monkeypatch):
    monkeypatch.setattr(
        document_upload, "run_query",
        lambda sql, params=None: [{"extraction_id": "mine", "candidate_json": {}}],
    )
    with pytest.raises(HTTPException) as excinfo:
        document_upload.publish_selected(
            batch_id="bat_1", extraction_ids=["someone-elses"], published_by="u1"
        )
    assert excinfo.value.status_code == 400


def test_missing_reader_dependency_gives_a_clear_message(monkeypatch):
    import importlib.util
    real = importlib.util.find_spec

    def missing_fitz(name, *a, **kw):
        return None if name == "fitz" else real(name, *a, **kw)

    monkeypatch.setattr(importlib.util, "find_spec", missing_fitz)
    with pytest.raises(HTTPException) as excinfo:
        document_upload._check_reader_available(".pdf")
    assert excinfo.value.status_code == 503
    assert "PyMuPDF" in excinfo.value.detail


def test_txt_upload_needs_no_extra_reader():
    document_upload._check_reader_available(".txt")   # must not raise
