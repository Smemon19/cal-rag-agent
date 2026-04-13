"""
Show a full snapshot of what is in the database.

Usage:
    python scripts/db_status.py
    python scripts/db_status.py --sample          # show sample rows from policies_v2
    python scripts/db_status.py --failed          # show why staged rows were rejected
"""

from __future__ import annotations

import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the policy database.")
    parser.add_argument("--sample", action="store_true", help="Print sample rows from policies_v2")
    parser.add_argument("--failed", action="store_true", help="Print staged rows that failed to publish")
    args = parser.parse_args()

    from policy_engine.db import run_query

    print("\n=== TABLE ROW COUNTS ===")
    try:
        counts = run_query(
            """
            SELECT 'policies_v2' AS tbl, COUNT(*) AS n FROM policies_v2
            UNION ALL SELECT 'policy_extractions_staging', COUNT(*) FROM policy_extractions_staging
            UNION ALL SELECT 'ingestion_batches', COUNT(*) FROM ingestion_batches
            UNION ALL SELECT 'documents', COUNT(*) FROM documents
            UNION ALL SELECT 'sections', COUNT(*) FROM sections
            """
        )
        for row in counts:
            print(f"  {row['tbl']}: {row['n']} rows")
    except Exception as e:
        print(f"  [error: {e}]")

    print("\n=== STAGING PIPELINE BREAKDOWN ===")
    try:
        staging = run_query(
            "SELECT status, COUNT(*) AS n FROM policy_extractions_staging GROUP BY status ORDER BY n DESC"
        )
        if staging:
            for row in staging:
                print(f"  {row['status']}: {row['n']}")
        else:
            print("  (no staged rows)")
    except Exception as e:
        print(f"  [error: {e}]")

    print("\n=== WHAT IS IN policies_v2 ===")
    try:
        total = run_query("SELECT COUNT(*) AS n FROM policies_v2")[0]["n"]
        print(f"  Total rows: {total}")

        topics = run_query(
            "SELECT DISTINCT topic FROM policies_v2 WHERE topic IS NOT NULL ORDER BY topic LIMIT 30"
        )
        if topics:
            print(f"\n  Topics ({len(topics)}):")
            for r in topics:
                print(f"    - {r['topic']}")
        else:
            print("\n  Topics: none")

        cats = run_query(
            "SELECT DISTINCT policy_category FROM policies_v2 WHERE policy_category IS NOT NULL ORDER BY policy_category LIMIT 20"
        )
        if cats:
            print(f"\n  Policy categories ({len(cats)}):")
            for r in cats:
                print(f"    - {r['policy_category']}")
        else:
            print("\n  Policy categories: none (column may be unpopulated)")

        depts = run_query(
            "SELECT DISTINCT department FROM policies_v2 WHERE department IS NOT NULL ORDER BY department LIMIT 20"
        )
        if depts:
            print(f"\n  Departments ({len(depts)}):")
            for r in depts:
                print(f"    - {r['department']}")

        statuses = run_query(
            "SELECT status, COUNT(*) AS n FROM policies_v2 GROUP BY status ORDER BY n DESC"
        )
        if statuses:
            print("\n  Status breakdown:")
            for r in statuses:
                print(f"    {r['status']}: {r['n']}")

    except Exception as e:
        print(f"  [error: {e}]")

    print("\n=== RECENT INGESTION BATCHES ===")
    try:
        batches = run_query(
            """
            SELECT batch_id, triggered_by, status, started_at, finished_at, result_summary
            FROM ingestion_batches ORDER BY started_at DESC LIMIT 10
            """
        )
        if batches:
            for b in batches:
                summary = ""
                if b.get("result_summary"):
                    try:
                        s = json.loads(b["result_summary"])
                        summary = f"published={s.get('published',0)} skipped={s.get('skipped_non_publishable',0)} failed={s.get('failed_publish',0)}"
                    except Exception:
                        summary = str(b["result_summary"])[:80]
                print(f"  [{b['status']}] {b['batch_id']}  started={b['started_at']}  {summary}")
        else:
            print("  (no batches found)")
    except Exception as e:
        print(f"  [error: {e}]")

    if args.sample:
        print("\n=== SAMPLE ROWS FROM policies_v2 ===")
        try:
            rows = run_query(
                """
                SELECT topic, subtopic, condition_text, action_text, source_quote, status
                FROM policies_v2 LIMIT 5
                """
            )
            if rows:
                for i, r in enumerate(rows, 1):
                    print(f"\n  Row {i}:")
                    for k, v in r.items():
                        if v:
                            print(f"    {k}: {str(v)[:120]}")
            else:
                print("  (no rows)")
        except Exception as e:
            print(f"  [error: {e}]")

    if args.failed:
        print("\n=== STAGED ROWS THAT FAILED OR WERE SKIPPED ===")
        try:
            rows = run_query(
                """
                SELECT extraction_id, status, publish_error, confidence,
                       candidate_json->>'chunk_type' AS chunk_type,
                       candidate_json->>'publishable' AS publishable,
                       candidate_json->>'reason_if_not_publishable' AS reason
                FROM policy_extractions_staging
                WHERE status IN ('failed_publish', 'skipped_non_publishable')
                ORDER BY created_at DESC LIMIT 20
                """
            )
            if rows:
                for r in rows:
                    print(f"\n  [{r['status']}] extraction_id={r['extraction_id']}")
                    print(f"    chunk_type={r['chunk_type']}  publishable={r['publishable']}  confidence={r['confidence']}")
                    if r.get("reason"):
                        print(f"    reason: {r['reason']}")
                    if r.get("publish_error"):
                        print(f"    error:  {r['publish_error']}")
            else:
                print("  (none)")
        except Exception as e:
            print(f"  [error: {e}]")

    print()


if __name__ == "__main__":
    main()
