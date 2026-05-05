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
    
    monkeypatch.setattr("adaptive_ingestion.admin_input_pipeline.run_query", mock_run_query)
    
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
