"""CLI entrypoint for adaptive policy ingestion pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from adaptive_ingestion.pipeline import AdaptiveIngestionPipeline, IngestionDocumentInput


def main() -> None:
    parser = argparse.ArgumentParser(description="Run adaptive ingestion on local policy files.")
    parser.add_argument("paths", nargs="+", help="Input file paths")
    parser.add_argument("--triggered-by", default="cli", help="Operator/user identifier")
    parser.add_argument("--apply-migrations", action="store_true", help="Apply approved and safe migrations")
    parser.add_argument("--publish", action="store_true", help="Replay + publish staged rows after migration")
    args = parser.parse_args()

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

    print(f"Batch: {batch_id}")
    print(f"Run: {result}")
    print(f"Schema proposals: {proposals}")

    if args.apply_migrations:
        mids = pipeline.apply_approved_migrations(applied_by=args.triggered_by)
        print(f"Migrations applied: {mids}")
    if args.publish:
        out = pipeline.replay_and_publish(batch_id=batch_id)
        print(f"Publish result: {out}")


if __name__ == "__main__":
    main()

