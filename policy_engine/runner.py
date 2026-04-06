"""
CLI harness for policy_engine: loops sample questions through answer_policy_question().

From repository root:
    python policy_engine/runner.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv

load_dotenv(_ROOT / ".env", override=True)

from policy_engine.service import answer_policy_question

SAMPLE_QUESTIONS = [
    "If I do internal marketing work, what do I bill it to?",
    "What do I bill client campaign work to?",
    "What's the difference between internal marketing and client marketing billing?",
    "Do I need approval for weekend overtime?",
    "What happens if payday falls on a holiday?",
]


def main() -> None:
    for i, question in enumerate(SAMPLE_QUESTIONS, start=1):
        print("=" * 72)
        print(f"[{i}] Question:\n  {question}\n")

        out = answer_policy_question(question)

        print("Search spec:")
        print(f"  {out['search_spec']}\n")

        print("SQL:")
        print(f"  {out['sql']}")
        print(f"  Params: {out['params']}\n")

        if out.get("used_relaxed_query"):
            print("Relaxed SQL (used):")
            print(f"  {out['relaxed_sql']}")
            print(f"  Params: {out['relaxed_params']}\n")

        print(f"Row count: {out['row_count']}\n")

        print("Answer:")
        print(f"  {out['answer']}\n")


if __name__ == "__main__":
    main()
