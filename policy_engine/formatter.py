"""
Final answer writer: grounded only on retrieved policy rows (no planning / no SQL).
"""

import json

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)

NO_POLICY_FOUND_TEXT = "No policy found in the retrieved data"


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
            "I found related policy records, but I could not find a direct policy "
            "that answers this exact question."
        )
    return (
        "I found related policy records, but I could not find a direct policy "
        "that answers this exact question. The closest relevant information says:\n"
        f"{summary}"
    )


def format_answer(question: str, rows: list[dict]) -> str:
    """
    Produce a concise answer from the user question and DB rows only.
    For comparison questions, multiple rows should be contrasted explicitly.
    """
    if not rows:
        return f"{NO_POLICY_FOUND_TEXT}."

    client = OpenAI()
    rows_json = json.dumps(rows, default=str, indent=2)

    system = (
        "You are a policy assistant. Answer the user's question using ONLY the evidence in the provided rows. "
        "Do not invent facts. Do not hallucinate rules or codes."
    )
    user_content = f"""User question:
{question}

Policy rows retrieved from the database (JSON):
{rows_json}

Instructions:
- Read every row carefully. For each row check these fields in order:
  1. action_text — the required action or rule
  2. condition_text — the condition or situation the rule applies to
  3. source_quote — verbatim text from the original document (most reliable for exact details like deadlines, amounts, codes)
- Answer the question directly using the most relevant row(s). Prefer source_quote for exact facts (dates, numbers, codes).
- If the answer spans multiple rows, synthesize them into one clear response.
- Never say "No policy found in the retrieved data" when rows are provided.
- If rows are related but do not directly answer the exact question, say that related policy records were found and summarize the closest information.
- If bill_to or charge_code appear and are relevant, mention them.
- If approval_required or approver appear and are relevant, mention them.
- If exception_text appears and is relevant, mention it.
- For comparison questions (A vs B), contrast the relevant rows clearly.
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
    return answer
