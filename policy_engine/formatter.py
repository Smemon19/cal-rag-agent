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
- Only say "No policy found in the retrieved data" if you have read every row and none of them contain ANY information relevant to the question — this should be rare when rows are provided.
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
    return (response.choices[0].message.content or "").strip()
