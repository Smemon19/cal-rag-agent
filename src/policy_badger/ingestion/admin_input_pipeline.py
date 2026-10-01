import datetime
import uuid
import json
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Dict

from policy_badger.ingestion.policy_extractor import PolicyExtractor
from policy_badger.ingestion.schema_dictionary import SchemaDictionary
from policy_badger.engine.formatter import format_answer
from policy_badger.engine.db import transaction

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

def _candidate_to_dict(candidate: Any) -> Dict[str, Any]:
    if hasattr(candidate, "model_dump"):  # Pydantic v2
        return candidate.model_dump()

    if hasattr(candidate, "dict"):  # Pydantic v1 fallback
        return candidate.dict()

    if is_dataclass(candidate):
        return asdict(candidate)

    if isinstance(candidate, dict):
        return candidate

    raise TypeError(
        f"candidate_json must be a pydantic model, dataclass, or dict, got {type(candidate).__name__}"
    )

def extract_submission(submission: AdminSubmission, use_llm: bool = False) -> None:
    schema_dict = SchemaDictionary()
    # Use deterministic extraction by default for speed/reliability in V1, unless LLM requested
    extractor = PolicyExtractor(schema_dictionary=schema_dict)
    extractor.use_llm = use_llm
    
    result = extractor.extract(
        document_id=submission.id,
        section_id="admin_input",
        section_text=submission.raw_text
    )
    
    # Store the candidate as a dict
    candidate_dict = _candidate_to_dict(result.payload.candidate_json)
    
    submission.extracted_json = {"item": candidate_dict}
    submission.extracted_json["meta"] = {
        "requested_llm": use_llm,
        "used_llm": use_llm and not getattr(extractor, "fallback_used", False),
        "fallback_used": getattr(extractor, "fallback_used", False),
        "fallback_reason": getattr(extractor, "fallback_reason", None)
    }
    
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
        
    values = {field: item.get(field) for field in PUBLISHABLE_FIELDS}
    values["topic"] = values.get("topic") or "General"
    values["source_quote"] = values.get("source_quote") or submission.raw_text
    values["status"] = "active"
    published_at = datetime.datetime.now()

    # The policy row and its audit row commit together or not at all. The
    # insert previously went through run_query, which never commits, so the
    # policy was silently rolled back.
    with transaction() as cur:
        cur.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = 'public' AND table_name = 'policies_v2'"
        )
        table_columns = {row["column_name"] for row in cur.fetchall()}
        columns = [c for c in values if c in table_columns and values[c] not in (None, "")]
        cur.execute(
            "INSERT INTO policies_v2 ({}) VALUES ({}) RETURNING policy_id".format(
                ", ".join(f'"{c}"' for c in columns), ", ".join(["%s"] * len(columns))
            ),
            tuple(values[c] for c in columns),
        )
        policy_id = cur.fetchone()["policy_id"]

        cur.execute(_AUDIT_SQL, (
            submission.id,
            str(policy_id),
            submission.submitted_by,
            published_by,
            submission.raw_text,
            json.dumps(submission.extracted_json),
            json.dumps(submission.extracted_json),
            submission.source_type,
            submission.version,
            submission.replaces_policy_id,
            submission.created_at,
            published_at,
        ))

    # Only mark the submission published once the commit has succeeded, so a
    # failed attempt can be retried.
    submission.status = "published"
    submission.published_at = published_at
    submission.published_by = published_by
    submission.published_policy_id = str(policy_id)
    return str(policy_id)


# Fields an admin can edit in Policy Studio; each is written when the live
# table has that column.
PUBLISHABLE_FIELDS = (
    "topic", "subtopic", "summary", "condition_text", "action_text",
    "approval_required", "approver", "exception_text", "source_quote",
)

_AUDIT_SQL = """
    INSERT INTO admin_policy_publish_audit (
        submission_id, published_policy_id, submitted_by, published_by,
        raw_text, extracted_json, final_json, source_type, version,
        replaces_policy_id, created_at, published_at
    ) VALUES (
        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
    )
"""
