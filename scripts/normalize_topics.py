"""
Topic Normalization Script
--------------------------
Run this AFTER ingesting all documents to clean up fragmented topic names.

What it does:
1. Reads every distinct topic currently in policies_v2
2. Sends the full list to GPT-4o-mini and asks it to propose canonical (clean) names
   by merging near-duplicates (e.g. "Timekeeping", "Time Tracking", "Time Charging Policy" → one name)
3. Prints the proposed mapping for your review
4. If you confirm, updates every row in policies_v2 to use the canonical names

Usage:
    python scripts/normalize_topics.py           # preview only
    python scripts/normalize_topics.py --apply   # preview + apply if you confirm
"""

from __future__ import annotations

import json
import os
import sys

from dotenv import load_dotenv

load_dotenv(override=True)

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from openai import OpenAI
from policy_engine.db import run_query
import psycopg2


def fetch_topics() -> list[tuple[str, int]]:
    """Return (topic, count) for every distinct non-null topic in policies_v2, sorted by count desc."""
    rows = run_query("""
        SELECT topic, COUNT(*) AS n
        FROM policies_v2
        WHERE topic IS NOT NULL AND topic != ''
        GROUP BY topic
        ORDER BY n DESC
    """)
    return [(r["topic"], int(r["n"])) for r in rows]


def propose_normalization(topics: list[tuple[str, int]]) -> dict[str, str]:
    """
    Ask GPT-4o-mini to propose a canonical name for each topic.
    Returns a dict of {original_topic: canonical_topic}.
    """
    # Format for display in prompt — counts help GPT understand which are dominant
    topic_list = "\n".join(f"- {t}  [{n} rows]" for t, n in topics)

    prompt = f"""You are cleaning up topic names in a policy database.

Below is a list of topic names that were automatically generated when policy documents were ingested.
Some of them mean the same thing but were labelled slightly differently (e.g. "Timekeeping", "Time Tracking", "Time Charging Policy").

Your job:
1. Group the topics that mean the same thing.
2. For each group, pick ONE clean, clear canonical name (Title Case, concise, 2-4 words).
3. Return a JSON object mapping every original topic name to its canonical name.
   Topics that are already clear and unique should map to themselves (or a slightly cleaner version).

Rules:
- Every topic in the input list must appear as a key in the output JSON. Use the EXACT topic name as the key — do NOT include the row count in the key.
- Canonical names should be short and human-readable.
- Do not invent new topics — only consolidate existing ones.
- Return JSON only, no markdown.

Topic list:
{topic_list}

Return format:
{{
  "original topic name": "Canonical Name",
  ...
}}"""

    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        temperature=0.0,
        messages=[{"role": "user", "content": prompt}],
    )
    content = response.choices[0].message.content or ""
    return json.loads(content)


def apply_normalization(mapping: dict[str, str]) -> int:
    """
    Update policies_v2 rows in place using the mapping.
    Returns the number of rows updated.
    """
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST", "127.0.0.1"),
        port=int(os.getenv("DB_PORT", "5432")),
        dbname=os.getenv("DB_NAME", "policy-badger"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", ""),
    )
    cur = conn.cursor()
    updated = 0
    for original, canonical in mapping.items():
        if original == canonical:
            continue
        cur.execute(
            "UPDATE policies_v2 SET topic = %s WHERE topic = %s",
            (canonical, original),
        )
        updated += cur.rowcount
    conn.commit()
    conn.close()
    return updated


def main() -> None:
    apply = "--apply" in sys.argv

    print("Fetching topics from policies_v2...")
    topics = fetch_topics()

    if not topics:
        print("No topics found in policies_v2. Ingest some documents first.")
        return

    print(f"Found {len(topics)} distinct topics across {sum(n for _, n in topics)} rows.\n")

    print("Asking GPT-4o-mini to propose canonical names...")
    mapping = propose_normalization(topics)

    # Show the proposed changes
    changes = {orig: canon for orig, canon in mapping.items() if orig != canon}
    no_changes = {orig: canon for orig, canon in mapping.items() if orig == canon}

    print("\n=== PROPOSED TOPIC MERGES ===")
    if changes:
        # Group by canonical name to show merges clearly
        by_canonical: dict[str, list[str]] = {}
        for orig, canon in changes.items():
            by_canonical.setdefault(canon, []).append(orig)
        for canon, originals in sorted(by_canonical.items()):
            print(f"\n  '{canon}'  ←  merges:")
            for o in originals:
                count = next((n for t, n in topics if t == o), 0)
                print(f"    - '{o}'  ({count} rows)")
    else:
        print("  No merges needed — all topics are already clean.")

    print(f"\n=== TOPICS KEPT AS-IS ({len(no_changes)}) ===")
    for orig in sorted(no_changes):
        count = next((n for t, n in topics if t == orig), 0)
        print(f"  - '{orig}'  ({count} rows)")

    if not apply:
        print("\nThis was a preview. Run with --apply to write changes to the database.")
        return

    if not changes:
        print("\nNothing to apply.")
        return

    confirm = input(f"\nApply {len(changes)} topic renames to the database? [y/N]: ").strip().lower()
    if confirm != "y":
        print("Aborted.")
        return

    updated = apply_normalization(mapping)
    print(f"\nDone. {updated} rows updated.")


if __name__ == "__main__":
    main()
