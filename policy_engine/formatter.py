"""
Final answer writer: grounded only on retrieved policy rows (no planning / no SQL).
"""

import json

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)


def format_answer(question: str, rows: list[dict]) -> str:
    """
    Produce a concise answer from the user question and DB rows only.
    For comparison questions, multiple rows should be contrasted explicitly.
    """
    client = OpenAI()
    rows_json = json.dumps(rows, default=str, indent=2)

    system = (
        "You are a policy assistant. Follow instructions exactly. Use only the provided JSON rows "
        "as evidence; if information is missing from the rows, say so. Do not invent policies or codes."
    )
    user_content = f"""User question:
{question}

Structured policy rows (JSON):
{rows_json}

Instructions:
- Answer ONLY using the provided rows. Do NOT hallucinate.
- If there are no rows or the JSON is empty, reply with exactly: No policy found
- Be concise but useful.
- If bill_to appears and is relevant, mention the billing destination.
- If charge_code appears and is relevant, mention the charge code.
- If approval_required or approver appear and are relevant, mention approval requirements.
- If exception_text appears and is relevant, mention exceptions.
- If multiple rows are present and the question compares two or more options, compare them clearly
  (what differs in billing, approval, exceptions, etc.).
- If the question implies a comparison but only one applicable row is present, state clearly that
  the other side was not found in the retrieved data.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ],
        temperature=0.2,
    )
    text = (response.choices[0].message.content or "").strip()
    if rows and text.lower() == "no policy found":
        snippets = []
        for r in rows[:3]:
            topic = r.get("topic") or r.get("subtopic") or "policy"
            cond = r.get("condition_text") or ""
            act = r.get("action_text") or ""
            quote = r.get("source_quote") or ""
            summary = " ".join(x for x in [str(topic), str(cond), str(act)] if x).strip()
            snippets.append(summary or str(quote))
        joined = "; ".join(s for s in snippets if s)
        return joined[:700] if joined else "No policy found"
    return text
