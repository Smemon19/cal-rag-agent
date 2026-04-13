"""CLI entrypoint for adaptive policy ingestion pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path


def _print_status() -> None:
    """Print a summary of what is currently in the database."""
    from policy_engine.db import run_query

    print("\n=== DATABASE STATUS ===\n")

    try:
        counts = run_query(
            """
            SELECT 'policies_v2' AS tbl, COUNT(*) AS n FROM policies_v2
            UNION ALL SELECT 'policy_extractions_staging', COUNT(*) FROM policy_extractions_staging
            UNION ALL SELECT 'ingestion_batches', COUNT(*) FROM ingestion_batches
            """
        )
        for row in counts:
            print(f"  {row['tbl']}: {row['n']} rows")
    except Exception as e:
        print(f"  [error reading table counts: {e}]")

    print()
    try:
        staging = run_query(
            "SELECT status, COUNT(*) AS n FROM policy_extractions_staging GROUP BY status ORDER BY n DESC"
        )
        if staging:
            print("Staging breakdown:")
            for row in staging:
                print(f"  {row['status']}: {row['n']}")
        else:
            print("Staging: empty")
    except Exception as e:
        print(f"  [error reading staging: {e}]")

    print()
    try:
        topics = run_query(
            "SELECT DISTINCT topic FROM policies_v2 WHERE topic IS NOT NULL ORDER BY topic LIMIT 30"
        )
        cats = run_query(
            "SELECT DISTINCT policy_category FROM policies_v2 WHERE policy_category IS NOT NULL ORDER BY policy_category LIMIT 20"
        )
        if topics:
            print("Topics in policies_v2:")
            for r in topics:
                print(f"  - {r['topic']}")
        else:
            print("Topics in policies_v2: none")
        if cats:
            print("Categories in policies_v2:")
            for r in cats:
                print(f"  - {r['policy_category']}")
    except Exception as e:
        print(f"  [error reading policies_v2 values: {e}]")

    print()
    try:
        batches = run_query(
            """
            SELECT batch_id, triggered_by, status, started_at, finished_at, result_summary
            FROM ingestion_batches ORDER BY started_at DESC LIMIT 5
            """
        )
        if batches:
            print("Recent ingestion batches:")
            for b in batches:
                summary = b.get("result_summary") or ""
                print(f"  [{b['status']}] {b['batch_id']} by {b['triggered_by']} at {b['started_at']}  {summary}")
        else:
            print("No ingestion batches found.")
    except Exception as e:
        print(f"  [error reading batches: {e}]")

    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run adaptive ingestion on local policy files.")
    parser.add_argument("paths", nargs="*", help="Input file paths (omit when using --status)")
    parser.add_argument("--triggered-by", default="cli", help="Operator/user identifier")
    parser.add_argument("--no-apply-migrations", action="store_true", help="Skip applying schema migrations")
    parser.add_argument("--no-publish", action="store_true", help="Skip publishing staged rows to policies_v2")
    parser.add_argument("--status", action="store_true", help="Show DB status and exit (no ingestion)")
    args = parser.parse_args()

    if args.status:
        _print_status()
        return

    if not args.paths:
        parser.error("Provide at least one file path, or use --status to inspect the database.")

    from adaptive_ingestion.pipeline import AdaptiveIngestionPipeline, IngestionDocumentInput

    docs = []
    for p in args.paths:
        path = Path(p)
        content = ""
        if path.suffix.lower() != ".pdf":
            content = path.read_text(encoding="utf-8")
        docs.append(
            IngestionDocumentInput(
                source_uri=f"file://{path.resolve()}",
                content=content,
                title=path.name,
            )
        )

    pipeline = AdaptiveIngestionPipeline()
    batch_id = pipeline.create_batch(triggered_by=args.triggered_by, trigger_source="cli")
    result = pipeline.run_batch(batch_id=batch_id, documents=docs)
    proposals = pipeline.plan_schema_changes(batch_id=batch_id)

    print(f"\nBatch:    {batch_id}")
    print(f"Run:      {result}")
    print(f"Proposals: {proposals}")

    if not args.no_apply_migrations:
        mids = pipeline.apply_approved_migrations(applied_by=args.triggered_by)
        print(f"Migrations applied: {mids}")

    if not args.no_publish:
        out = pipeline.replay_and_publish(batch_id=batch_id)
        pub = out.get("published", 0)
        skipped = out.get("summary", {}).get("skipped_non_publishable", 0)
        failed = out.get("summary", {}).get("failed_publish", 0)
        print(f"\nPublish result: {pub} published, {skipped} skipped, {failed} failed")
        if pub == 0:
            print("  Tip: run with --status to inspect what is in the database.")


if __name__ == "__main__":
    main()

