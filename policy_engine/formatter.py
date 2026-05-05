"""
Final answer writer: grounded only on retrieved policy rows (no planning / no SQL).
"""

import json

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)

NO_POLICY_FOUND_TEXT = "No policy found in the retrieved data"
DIRECT_ANSWER_HEADER = "Direct Answer"
KEY_DETAILS_HEADER = "Key Details"
IMPORTANT_NOTES_HEADER = "Important Notes / Exceptions"
ACTION_GUIDANCE_HEADER = "What You Should Do"


def _clean_text(value: object) -> str:
    return " ".join(str(value or "").split())


def _unique_values(rows: list[dict], field: str, limit: int = 3) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for row in rows:
        text = _clean_text(row.get(field))
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        values.append(text)
        seen.add(key)
        if len(values) >= limit:
            break
    return values


def _first_field_value(rows: list[dict], field: str) -> object | None:
    for row in rows:
        value = row.get(field)
        if value not in (None, ""):
            return value
    return None


def _approval_detail(rows: list[dict]) -> str | None:
    approval_required = _first_field_value(rows, "approval_required")
    if approval_required in (None, ""):
        return None

    approver = _clean_text(_first_field_value(rows, "approver"))
    if approval_required is True:
        detail = "Approval required: Yes."
        if approver:
            detail += f" Approver: {approver}."
        return detail
    if approval_required is False:
        return "Approval required: No, based on the retrieved policy rows."
    return f"Approval required: {_clean_text(approval_required)}"


def _format_bullets(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def _has_required_structure(answer: str) -> bool:
    lowered = answer.lower()
    required_headers = (DIRECT_ANSWER_HEADER, KEY_DETAILS_HEADER, ACTION_GUIDANCE_HEADER)
    return all(header.lower() in lowered for header in required_headers)


def _direct_answer_from_model(answer: str) -> str:
    lines = [line.strip(" -") for line in answer.splitlines() if line.strip()]
    if not lines:
        return "I found relevant policy information, but the retrieved rows do not support a more specific answer."
    direct = lines[0]
    if len(lines) > 1 and len(direct) < 80 and not lines[1].endswith(":"):
        direct = f"{direct} {lines[1]}"
    return direct


def _structured_answer_from_model(answer: str, rows: list[dict]) -> str:
    if _has_required_structure(answer):
        return answer

    key_details: list[str] = []
    approval = _approval_detail(rows)
    if approval:
        key_details.append(approval)
    for condition in _unique_values(rows, "condition_text", limit=2):
        key_details.append(f"Applies when: {condition}")
    for bill_to in _unique_values(rows, "bill_to", limit=1):
        key_details.append(f"Bill to: {bill_to}")
    for charge_code in _unique_values(rows, "charge_code", limit=1):
        key_details.append(f"Charge code: {charge_code}")
    if not key_details:
        topics = []
        for row in rows[:2]:
            topic = " / ".join(
                part for part in (_clean_text(row.get("topic")), _clean_text(row.get("subtopic"))) if part
            )
            if topic:
                topics.append(topic)
        key_details = [f"Relevant policy area: {', '.join(topics)}."] if topics else [
            "The retrieved policy rows do not include additional conditions."
        ]

    notes = [f"Exception: {text}" for text in _unique_values(rows, "exception_text", limit=2)]
    actions = _unique_values(rows, "action_text", limit=2)
    if not actions:
        actions = ["Follow the policy guidance above, and confirm the details before submitting time or expenses."]

    sections = [
        f"{DIRECT_ANSWER_HEADER}\n{_direct_answer_from_model(answer)}",
        f"{KEY_DETAILS_HEADER}\n{_format_bullets(key_details)}",
    ]
    if notes:
        sections.append(f"{IMPORTANT_NOTES_HEADER}\n{_format_bullets(notes)}")
    sections.append(f"{ACTION_GUIDANCE_HEADER}\n{_format_bullets(actions)}")
    return "\n\n".join(sections)


def _summarize_related_rows(rows: list[dict], limit: int = 3) -> str:
    snippets = []
    for row in rows[:limit]:
        topic_parts = [str(row.get(k) or "").strip() for k in ("topic", "subtopic")]
        topic = " / ".join(part for part in topic_parts if part)
        evidence = (
            row.get("action_text")
            or row.get("condition_text")
            or row.get("summary")
            or row.get("source_quote")
            or ""
        )
        evidence_text = str(evidence).strip()
        if len(evidence_text) > 260:
            evidence_text = evidence_text[:257].rstrip() + "..."
        if topic and evidence_text:
            snippets.append(f"- {topic}: {evidence_text}")
        elif evidence_text:
            snippets.append(f"- {evidence_text}")

    if not snippets:
        return ""
    return "\n".join(snippets)


def _related_records_answer(rows: list[dict]) -> str:
    summary = _summarize_related_rows(rows)
    if not summary:
        return (
            f"{DIRECT_ANSWER_HEADER}\n"
            "I found related policy records, but not a direct answer to this exact question.\n\n"
            f"{KEY_DETAILS_HEADER}\n"
            "- The retrieved rows do not provide enough detail to answer this directly.\n\n"
            f"{ACTION_GUIDANCE_HEADER}\n"
            "- Use the related policy context carefully and confirm before taking action."
        )
    return (
        f"{DIRECT_ANSWER_HEADER}\n"
        "I found related policy records, but not a direct answer to this exact question.\n\n"
        f"{KEY_DETAILS_HEADER}\n"
        f"{summary}\n\n"
        f"{ACTION_GUIDANCE_HEADER}\n"
        "- Use the related policy context carefully and confirm before taking action."
    )


def format_answer(question: str, rows: list[dict], intent: str | None = None) -> str:
    """
    Produce a concise, structured answer from the user question and DB rows only.
    For comparison questions, multiple rows should be contrasted explicitly.
    """
    if not rows:
        return f"{NO_POLICY_FOUND_TEXT}."

    client = OpenAI()
    rows_json = json.dumps(rows, default=str, indent=2)

    system = (
        "You are a policy assistant. Answer the user's question using ONLY the evidence in the provided rows. "
        "Do not invent facts. Do not hallucinate rules or codes. Be clear, practical, and slightly conversational."
    )
    intent_instruction = ""
    if intent == "expense_category":
        intent_instruction = "\n- INTENT: The user is asking about an expense category. Direct Answer should emphasize the category FIRST, then include approval/process."
    elif intent == "approval_requirement":
        intent_instruction = "\n- INTENT: The user is asking about an approval requirement. Direct Answer should be YES/NO first, then include details."
    elif intent == "complete_list":
        intent_instruction = "\n- INTENT: The user is asking for a complete list. Return the full list cleanly, and avoid unnecessary extra explanation."
    elif intent == "definition":
        intent_instruction = "\n- INTENT: The user is asking for a definition. Give a clean definition first, with minimal extra detail."
    elif intent == "procedure_steps":
        intent_instruction = "\n- INTENT: The user is asking for procedure steps. Focus on step-by-step instructions in the response."

    user_content = f"""User question:
{question}

Policy rows retrieved from the database (JSON):
{rows_json}

Instructions:
- Read every row carefully. For each row check these fields in order:
  1. action_text — the required action or rule
  2. condition_text — the condition or situation the rule applies to
  3. source_quote — verbatim text from the original document (most reliable for exact details like deadlines, amounts, codes)
- Use this exact structure:
  Direct Answer
  <1-2 clear lines that directly answer the employee's question>

  Key Details
  - <conditions, limits, amounts, dates, bill_to, charge_code, or other important details>

  Important Notes / Exceptions
  - <exceptions, rare cases, caveats, or approval nuance; omit this section only if none apply>

  What You Should Do
  - <specific action guidance for the employee>{intent_instruction}
- Answer the question directly using the most relevant row(s). Prefer source_quote for exact facts (dates, numbers, codes), but do not dump raw policy text.
- If the answer spans multiple rows, synthesize them into one clear response.
- Never say "No policy found in the retrieved data" when rows are provided.
- If rows are related but do not directly answer the exact question, say that related policy records were found and summarize the closest information.
- If bill_to or charge_code appear and are relevant, mention them.
- If approval_required or approver appear and are relevant, emphasize approval clearly.
- If exception_text appears and is relevant, move it into "Important Notes / Exceptions".
- If condition_text appears and is relevant, include it in "Key Details".
- If action_text appears and is relevant, include it in "What You Should Do".
- For comparison questions (A vs B), contrast the relevant rows clearly.
- Keep the answer useful and concise. Avoid repeated points, robotic phrasing, and vague language like "may" unless the evidence explains the condition.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ],
        temperature=0.1,
    )
    answer = (response.choices[0].message.content or "").strip()
    if NO_POLICY_FOUND_TEXT.lower() in answer.lower():
        return _related_records_answer(rows)
    return _structured_answer_from_model(answer, rows)
