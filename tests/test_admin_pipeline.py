import pytest
from policy_badger.ingestion.admin_input_pipeline import (
    AdminSubmission,
    create_submission,
    extract_submission,
    validate_submission,
    preview_submission,
    publish_submission
)
from contextlib import contextmanager


class FakeCursor:
    """Records statements run inside admin_input_pipeline.transaction()."""

    def __init__(self, policy_id="test_id_123", columns=None, fail_on=None):
        self.policy_id = policy_id
        self.columns = columns or [
            "policy_id", "topic", "subtopic", "summary", "condition_text", "action_text",
            "approval_required", "approver", "exception_text", "source_quote", "status",
        ]
        self.fail_on = fail_on
        self.statements = []
        self.committed = False
        self._last = None

    def execute(self, sql, params=None):
        if self.fail_on and self.fail_on in sql:
            raise RuntimeError("boom")
        self.statements.append((sql, params))
        self._last = sql

    def fetchall(self):
        return [{"column_name": c} for c in self.columns]

    def fetchone(self):
        return {"policy_id": self.policy_id}


def use_fake_transaction(monkeypatch, cursor):
    @contextmanager
    def fake_transaction():
        yield cursor
        cursor.committed = True

    monkeypatch.setattr("policy_badger.ingestion.admin_input_pipeline.transaction", fake_transaction)

def test_extraction_works(monkeypatch):
    sub = create_submission("Test Policy", "If you work late, charge dinner to 1234.")
    
    from dataclasses import dataclass
    
    @dataclass
    class MockCandidateJson:
        topic: str = "Meals"
        action_text: str = "Charge dinner to 1234."
        condition_text: str = "If you work late"
        source_quote: str = "If you work late, charge dinner to 1234."

    class MockPayload:
        def __init__(self):
            self.candidate_json = MockCandidateJson()

    class MockResult:
        def __init__(self):
            self.payload = MockPayload()

    class MockExtractor:
        def __init__(self, **kwargs):
            self.use_llm = False
        def extract(self, **kwargs):
            return MockResult()

    monkeypatch.setattr("policy_badger.ingestion.admin_input_pipeline.PolicyExtractor", MockExtractor)

    extract_submission(sub)
    
    assert sub.status == "extracted"
    assert "item" in sub.extracted_json
    assert sub.extracted_json["item"]["topic"] == "Meals"

def test_extraction_works_with_dict(monkeypatch):
    sub = create_submission("Test Policy", "If you work late, charge dinner to 1234.")
    
    class MockPayload:
        def __init__(self):
            self.candidate_json = {
                "topic": "Meals",
                "action_text": "Charge dinner to 1234.",
                "condition_text": "If you work late",
                "source_quote": "If you work late, charge dinner to 1234."
            }

    class MockResult:
        def __init__(self):
            self.payload = MockPayload()

    class MockExtractor:
        def __init__(self, **kwargs):
            self.use_llm = False
        def extract(self, **kwargs):
            return MockResult()

    monkeypatch.setattr("policy_badger.ingestion.admin_input_pipeline.PolicyExtractor", MockExtractor)

    extract_submission(sub)
    
    assert sub.status == "extracted"
    assert "item" in sub.extracted_json
    assert sub.extracted_json["item"]["topic"] == "Meals"

def test_validation_works():
    # Valid
    sub1 = AdminSubmission(id="1", title="T", raw_text="R")
    sub1.extracted_json = {"item": {"topic": "T", "action_text": "A", "source_quote": "S"}}
    sub1.confidence = 0.8
    validate_submission(sub1)
    assert sub1.status == "extracted"

    # Missing quote
    sub2 = AdminSubmission(id="2", title="T", raw_text="R")
    sub2.extracted_json = {"item": {"topic": "T", "action_text": "A"}}
    sub2.confidence = 0.8
    validate_submission(sub2)
    assert sub2.status == "needs_review"

def test_preview_works(monkeypatch):
    sub = AdminSubmission(id="1", title="T", raw_text="R")
    sub.extracted_json = {"item": {"topic": "Meals", "action_text": "Eat", "source_quote": "Q"}}
    
    # Mock format_answer
    monkeypatch.setattr("policy_badger.ingestion.admin_input_pipeline.format_answer", lambda q, r: f"Mock Answer for {q}")
    
    preview = preview_submission(sub, "test question")
    assert preview["simulated_answer"] == "Mock Answer for test question"

def test_publish_inserts_correctly(monkeypatch):
    sub = AdminSubmission(id="1", title="T", raw_text="R", status="extracted")
    sub.extracted_json = {"item": {
        "topic": "Meals", "action_text": "Eat", "source_quote": "Q",
        "approver": "Finance", "approval_required": True, "unknown_field": "x",
    }}
    cur = FakeCursor()
    use_fake_transaction(monkeypatch, cur)

    policy_id = publish_submission(sub)

    insert_sql, insert_params = next(st for st in cur.statements if "INSERT INTO policies_v2" in st[0])
    assert policy_id == "test_id_123"
    assert cur.committed
    # Every Studio-editable field present on the table is written.
    for column, value in {"topic": "Meals", "action_text": "Eat", "approver": "Finance", "approval_required": True}.items():
        assert f'"{column}"' in insert_sql
        assert value in insert_params
    assert "unknown_field" not in insert_sql
    assert sub.status == "published"


def test_publish_skips_columns_missing_from_table(monkeypatch):
    sub = AdminSubmission(id="1", title="T", raw_text="R", status="extracted")
    sub.extracted_json = {"item": {"topic": "Meals", "action_text": "Eat", "approver": "Finance"}}
    cur = FakeCursor(columns=["policy_id", "topic", "action_text", "source_quote", "status"])
    use_fake_transaction(monkeypatch, cur)

    publish_submission(sub)

    insert_sql = next(st[0] for st in cur.statements if "INSERT INTO policies_v2" in st[0])
    assert '"approver"' not in insert_sql


def test_failed_audit_leaves_submission_retryable(monkeypatch):
    sub = AdminSubmission(id="1", title="T", raw_text="R", status="extracted")
    sub.extracted_json = {"item": {"topic": "Meals", "action_text": "Eat", "source_quote": "Q"}}
    cur = FakeCursor(fail_on="admin_policy_publish_audit")
    use_fake_transaction(monkeypatch, cur)

    with pytest.raises(RuntimeError):
        publish_submission(sub)

    assert not cur.committed
    assert sub.status == "extracted"
    assert sub.published_policy_id is None

def test_vague_input_triggers_clarification():
    sub = AdminSubmission(id="1", title="T", raw_text="Just do it.")
    sub.extracted_json = {"item": {"topic": "General"}}
    sub.confidence = 0.2
    
    validate_submission(sub)
    
    assert sub.status == "needs_clarification"

def test_clear_input_does_not_trigger_clarification():
    sub = AdminSubmission(id="1", title="T", raw_text="Approval required. $50 max.")
    sub.extracted_json = {"item": {"topic": "T", "action_text": "A"}}
    sub.confidence = 0.8
    
    validate_submission(sub)
    
    assert sub.status != "needs_clarification"

def test_clarification_improves_confidence(monkeypatch):
    from policy_badger.ingestion.admin_input_pipeline import generate_clarification_questions
    
    sub = AdminSubmission(id="1", title="T", raw_text="Vague policy.")
    sub.extracted_json = {"item": {"topic": "T"}}
    
    qs = generate_clarification_questions(sub)
    assert len(qs) >= 2
    
    # Simulate user answering
    sub.raw_text += " Clarification: Needs approval by manager. Maximum amount is $50."
    
    # Re-extract
    from dataclasses import dataclass
    
    class MockResult:
        class Payload:
            @dataclass
            class CandidateJson:
                topic: str = "T"
                action_text: str = "A"
                condition_text: str = "C"
                source_quote: str = "Q"
            candidate_json = CandidateJson()
        payload = Payload()

    class MockExtractor:
        def __init__(self, **kwargs):
            self.use_llm = False
        def extract(self, **kwargs):
            return MockResult()

    monkeypatch.setattr("policy_badger.ingestion.admin_input_pipeline.PolicyExtractor", MockExtractor)
    
    extract_submission(sub)
    
    # Now it has topic, action, condition, approval in text, digits in text.
    assert sub.confidence >= 0.8
    
    validate_submission(sub)
    assert sub.status == "extracted"

def test_create_submission_stores_audit_metadata():
    sub = create_submission("Test Audit", "Content", submitted_by="admin@example.com")
    assert sub.submitted_by == "admin@example.com"
    assert sub.created_at is not None
    assert sub.version == 1
    assert sub.source_type == "admin_entry"
    assert sub.replaces_policy_id is None

def test_publish_submission_stores_audit_record(monkeypatch):
    sub = AdminSubmission(
        id="audit_sub_1", 
        title="T", 
        raw_text="Audit text", 
        status="extracted",
        submitted_by="creator@example.com"
    )
    sub.extracted_json = {"item": {"topic": "Audit", "action_text": "Do audit", "source_quote": "Q"}}

    cur = FakeCursor(policy_id="new_policy_123")
    use_fake_transaction(monkeypatch, cur)

    # We can simulate replacing an older policy
    sub.replaces_policy_id = "old_policy_456"
    
    policy_id = publish_submission(sub, published_by="publisher@example.com")
    
    assert policy_id == "new_policy_123"
    assert sub.published_policy_id == "new_policy_123"
    assert sub.published_by == "publisher@example.com"
    assert sub.published_at is not None
    
    # The audit row is written in the same transaction as the policy.
    audit_sql, audit_args = cur.statements[-1]
    assert cur.committed
    
    assert "INSERT INTO admin_policy_publish_audit" in audit_sql
    # audit_params: (submission.id, submission.published_policy_id, submission.submitted_by, submission.published_by, submission.raw_text, json.dumps(submission.extracted_json), json.dumps(submission.extracted_json), submission.source_type, submission.version, submission.replaces_policy_id, submission.created_at, submission.published_at)
    assert audit_args[0] == "audit_sub_1"
    assert audit_args[1] == "new_policy_123"
    assert audit_args[2] == "creator@example.com"
    assert audit_args[3] == "publisher@example.com"
    assert audit_args[4] == "Audit text"
    assert '"topic": "Audit"' in audit_args[5] # JSON string of extracted_json
    assert audit_args[7] == "admin_entry"
    assert audit_args[8] == 1
    assert audit_args[9] == "old_policy_456"

def test_extraction_with_llm_flag(monkeypatch):
    from policy_badger.ingestion.admin_input_pipeline import extract_submission, create_submission
    
    sub = create_submission("LLM Policy", "Testing LLM flag.")
    
    called_use_llm = None
    
    class MockExtractor:
        def __init__(self, **kwargs):
            self.use_llm = False
            
        def extract(self, **kwargs):
            nonlocal called_use_llm
            called_use_llm = self.use_llm
            
            class MockPayload:
                candidate_json = {
                    "topic": "LLM Topic",
                    "action_text": "LLM Action",
                    "source_quote": "Testing LLM flag."
                }
            class MockResult:
                payload = MockPayload()
            return MockResult()

    monkeypatch.setattr("policy_badger.ingestion.admin_input_pipeline.PolicyExtractor", MockExtractor)

    extract_submission(sub, use_llm=True)
    
    assert called_use_llm is True
    assert sub.status == "extracted"
    assert sub.extracted_json["item"]["topic"] == "LLM Topic"
