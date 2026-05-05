import datetime
import uuid
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict

from adaptive_ingestion.policy_extractor import PolicyExtractor
from adaptive_ingestion.schema_dictionary import SchemaDictionary
from policy_engine.formatter import format_answer
from policy_engine.db import execute, run_query

@dataclass
class AdminSubmission:
    id: str
    title: str
    raw_text: str
    extracted_json: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    status: str = "draft"  # draft, extracted, needs_clarification, needs_review, published
    created_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    effective_date: datetime.datetime | None = None
    submitted_by: str | None = None
    reviewed_by: str | None = None
    published_by: str | None = None
    reviewed_at: datetime.datetime | None = None
    published_at: datetime.datetime | None = None
    source_type: str = "admin_entry"
    version: int = 1
    replaces_policy_id: str | None = None
    published_policy_id: str | None = None


_submissions: Dict[str, AdminSubmission] = {}

def create_submission(title: str, raw_text: str, submitted_by: str | None = None, effective_date: datetime.datetime | None = None) -> AdminSubmission:
    sub = AdminSubmission(
        id=str(uuid.uuid4()),
        title=title,
        raw_text=raw_text,
        submitted_by=submitted_by,
        effective_date=effective_date
    )
    _submissions[sub.id] = sub
    return sub

def extract_submission(submission: AdminSubmission) -> None:
    schema_dict = SchemaDictionary()
    # Force deterministic extraction by disabling LLM for speed/reliability in V1
    extractor = PolicyExtractor(schema_dictionary=schema_dict)
    extractor.use_llm = False
    
    result = extractor.extract(
        document_id=submission.id,
        section_id="admin_input",
        section_text=submission.raw_text
    )
    
    # Store the candidate as a dict
    candidate_dict = asdict(result.payload.candidate_json)
    
    submission.extracted_json = {"item": candidate_dict}
    
    # Calculate confidence scoring
    score = 0.0
    if candidate_dict.get("topic"): score += 0.2
    if candidate_dict.get("condition_text"): score += 0.2
    if candidate_dict.get("action_text"): score += 0.2
    if "approval" in submission.raw_text.lower() or candidate_dict.get("approval_required"): score += 0.2
    import re
    if re.search(r'\d+', submission.raw_text): score += 0.2
    
    submission.confidence = min(1.0, score)
    submission.status = "extracted"

def generate_clarification_questions(submission: AdminSubmission) -> list[str]:
    questions = []
    text = submission.raw_text.lower()
    item = submission.extracted_json.get("item", {})
    
    if "approval" not in text:
        questions.append("Does this policy require approval? If so, who approves it?")
    if not item.get("condition_text") or len(item.get("condition_text", "")) < 8:
        questions.append("What exactly are the conditions or limits? (e.g., after 8 PM, defining 'late')")
    import re
    if not re.search(r'\d+', text):
        questions.append("What is the maximum allowed amount or numeric limit?")
        
    return questions

def validate_submission(submission: AdminSubmission) -> None:
    item = submission.extracted_json.get("item")
    if not item:
        submission.status = "needs_clarification"
        return
    
    topic = item.get("topic")
    action = item.get("action_text")
    condition = item.get("condition_text")
    quote = item.get("source_quote")
    
    if submission.confidence < 0.7 or not topic or not action:
        submission.status = "needs_clarification"
        return
        
    if not quote or (not action and not condition):
        submission.status = "needs_review"
        return
        
    submission.status = "extracted"

def preview_submission(submission: AdminSubmission, question: str = "What is the policy regarding this?") -> Dict[str, Any]:
    item = submission.extracted_json.get("item")
    if not item:
        return {"error": "No extracted data to preview."}
        
    # Simulate DB row
    row = dict(item)
    answer = format_answer(question, [row])
    
    return {
        "extracted_json": submission.extracted_json,
        "sample_question": question,
        "simulated_answer": answer
    }

def publish_submission(submission: AdminSubmission, published_by: str | None = None) -> str:
    if submission.status not in ("extracted", "needs_review"):
        raise ValueError(f"Cannot publish submission in status {submission.status}")
        
    item = submission.extracted_json.get("item")
    if not item:
        raise ValueError("No extracted data to publish")
        
    topic = item.get("topic") or "General"
    subtopic = item.get("subtopic")
    condition_text = item.get("condition_text")
    action_text = item.get("action_text")
    source_quote = item.get("source_quote") or submission.raw_text
    
    sql = """
        INSERT INTO policies_v2 (
            topic, subtopic, condition_text, action_text, source_quote, status
        ) VALUES (
            %s, %s, %s, %s, %s, 'active'
        ) RETURNING policy_id
    """
    params = (topic, subtopic, condition_text, action_text, source_quote)
    
    # Simulate insert if we can't get returning id natively from generic execute
    # Actually, execute in db.py returns rowcount for INSERT. 
    # To get RETURNING we need to use run_query
    rows = run_query(sql, params)
    policy_id = rows[0]["policy_id"] if rows else "unknown"
    
    submission.status = "published"
    submission.published_at = datetime.datetime.now()
    submission.published_by = published_by
    submission.published_policy_id = str(policy_id)
    
    # Insert audit record
    audit_sql = """
        INSERT INTO admin_policy_publish_audit (
            submission_id, published_policy_id, submitted_by, published_by,
            raw_text, extracted_json, final_json, source_type, version,
            replaces_policy_id, created_at, published_at
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        )
    """
    audit_params = (
        submission.id,
        submission.published_policy_id,
        submission.submitted_by,
        submission.published_by,
        submission.raw_text,
        json.dumps(submission.extracted_json),
        json.dumps(submission.extracted_json),
        submission.source_type,
        submission.version,
        submission.replaces_policy_id,
        submission.created_at,
        submission.published_at
    )
    execute(audit_sql, audit_params)
    
    return str(policy_id)
