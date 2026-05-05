import pytest
from adaptive_ingestion.admin_input_pipeline import (
    AdminSubmission,
    create_submission,
    extract_submission,
    validate_submission,
    preview_submission,
    publish_submission
)

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

    monkeypatch.setattr("adaptive_ingestion.admin_input_pipeline.PolicyExtractor", MockExtractor)

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
    monkeypatch.setattr("adaptive_ingestion.admin_input_pipeline.format_answer", lambda q, r: f"Mock Answer for {q}")
    
    preview = preview_submission(sub, "test question")
    assert preview["simulated_answer"] == "Mock Answer for test question"

def test_publish_inserts_correctly(monkeypatch):
    sub = AdminSubmission(id="1", title="T", raw_text="R", status="extracted")
    sub.extracted_json = {"item": {"topic": "Meals", "action_text": "Eat", "source_quote": "Q"}}

    # Mock DB run_query
    called_sql = ""
    called_params = ()
    def mock_run_query(sql, params):
        nonlocal called_sql, called_params
        called_sql = sql
        called_params = params
        return [{"policy_id": "test_id_123"}]
        
    def mock_execute(sql, params):
        pass
    
    monkeypatch.setattr("adaptive_ingestion.admin_input_pipeline.run_query", mock_run_query)
    monkeypatch.setattr("adaptive_ingestion.admin_input_pipeline.execute", mock_execute)
    
    policy_id = publish_submission(sub)
    
    assert policy_id == "test_id_123"
    assert "INSERT INTO policies_v2" in called_sql
    assert called_params[0] == "Meals"
    assert called_params[3] == "Eat"
    assert sub.status == "published"

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
    from adaptive_ingestion.admin_input_pipeline import generate_clarification_questions
    
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

    monkeypatch.setattr("adaptive_ingestion.admin_input_pipeline.PolicyExtractor", MockExtractor)
    
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

    called_sqls = []
    called_params = []
    
    def mock_run_query(sql, params):
        called_sqls.append(sql)
        called_params.append(params)
        return [{"policy_id": "new_policy_123"}]
        
    def mock_execute(sql, params):
        called_sqls.append(sql)
        called_params.append(params)
        return 1

    monkeypatch.setattr("adaptive_ingestion.admin_input_pipeline.run_query", mock_run_query)
    monkeypatch.setattr("adaptive_ingestion.admin_input_pipeline.execute", mock_execute)
    
    # We can simulate replacing an older policy
    sub.replaces_policy_id = "old_policy_456"
    
    policy_id = publish_submission(sub, published_by="publisher@example.com")
    
    assert policy_id == "new_policy_123"
    assert sub.published_policy_id == "new_policy_123"
    assert sub.published_by == "publisher@example.com"
    assert sub.published_at is not None
    
    # Verify execute was called for the audit log
    audit_sql = called_sqls[1]
    audit_args = called_params[1]
    
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
