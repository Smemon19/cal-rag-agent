"""
Admin document upload: file in, reviewed policies out.

The Policy Studio could only take pasted text for a single policy. This drives
the existing adaptive pipeline from an uploaded .pdf / .docx / .txt, then hands
the extracted candidates to dedup.review_batch so the admin sees which are new
and which the database already holds. Nothing is published until the admin
selects rows and calls publish_selected.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Optional
from uuid import uuid4

from fastapi import HTTPException

from policy_badger.engine.db import execute, run_query
from policy_badger.ingestion import dedup
from policy_badger.ingestion.pipeline import AdaptiveIngestionPipeline, IngestionDocumentInput
from policy_badger.ingestion.publisher import NonPublishableChunk

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".pdf", ".docx", ".txt", ".md"}

# Large enough for a staff handbook, small enough to refuse a mistaken upload.
MAX_UPLOAD_BYTES = 25 * 1024 * 1024


# Reader dependencies per file type, so a missing one is reported as a clear
# message instead of surfacing "No module named 'fitz'" to an admin.
_READER_REQUIREMENTS = {
    ".pdf": (("fitz", "PyMuPDF"), ("pytesseract", "pytesseract")),
    ".docx": (("docx", "python-docx"),),
}


def _check_reader_available(suffix: str) -> None:
    import importlib.util

    for module, package in _READER_REQUIREMENTS.get(suffix, ()):
        if importlib.util.find_spec(module) is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    f"{suffix.upper().lstrip('.')} support is not installed on the server "
                    f"(missing '{package}'). Install it, or upload the policy as a TXT file."
                ),
            )


def _check_upload(filename: str, size: int) -> str:
    suffix = Path(filename or "").suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix or filename}'. Upload a PDF, DOCX, TXT or MD file.",
        )
    if size <= 0:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if size > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File is too large ({size // (1024 * 1024)}MB). The limit is {MAX_UPLOAD_BYTES // (1024 * 1024)}MB.",
        )
    return suffix


def ingest_upload(
    *,
    filename: str,
    content: bytes,
    title: Optional[str],
    triggered_by: str,
) -> dict[str, Any]:
    """
    Extract policies from an uploaded document and classify them for review.

    Returns the batch id plus the review payload. The staged rows stay in
    policy_extractions_staging with status 'pending' until published or
    discarded, so a half-finished review can be resumed.
    """
    suffix = _check_upload(filename, len(content))
    _check_reader_available(suffix)

    pipeline = AdaptiveIngestionPipeline()
    batch_id = pipeline.create_batch(triggered_by=triggered_by, trigger_source="admin_upload")

    # The PDF and DOCX readers work on real paths, so the bytes land in a
    # temporary directory that is removed as soon as extraction finishes.
    with TemporaryDirectory() as tmp:
        tmp_path = Path(tmp) / f"upload_{uuid4().hex}{suffix}"
        tmp_path.write_bytes(content)

        document = IngestionDocumentInput(
            source_uri=f"file://{tmp_path}",
            content="",
            title=title or Path(filename).stem,
            doc_type="admin_upload",
            version=None,
        )
        try:
            stats = pipeline.run_batch(batch_id=batch_id, documents=[document])
        except Exception as e:
            execute(
                "UPDATE ingestion_batches SET status = 'failed' WHERE batch_id = %s",
                (batch_id,),
            )
            logger.exception("Upload ingestion failed for batch %s", batch_id)
            raise HTTPException(status_code=422, detail=f"Could not read that document: {e}") from e

    review = get_review(batch_id)
    review["batch_id"] = batch_id
    review["filename"] = filename
    review["sections_total"] = stats.get("sections_total", 0)
    return review


def get_review(batch_id: str) -> dict[str, Any]:
    """Classification of everything still awaiting a decision in this batch."""
    rows = run_query(
        """
        SELECT extraction_id, candidate_json, confidence, status
        FROM policy_extractions_staging
        WHERE batch_id = %s AND status IN ('pending', 'blocked_schema')
        ORDER BY created_at ASC
        """,
        (batch_id,),
    )
    if not rows:
        batch = run_query("SELECT batch_id FROM ingestion_batches WHERE batch_id = %s", (batch_id,))
        if not batch:
            raise HTTPException(status_code=404, detail="Ingestion batch not found.")
    return dedup.review_batch(rows)


def publish_selected(
    *,
    batch_id: str,
    extraction_ids: list[str],
    published_by: str,
) -> dict[str, Any]:
    """
    Publish only the extractions the admin ticked; discard the rest.

    Mirrors AdaptiveIngestionPipeline.replay_and_publish, but scoped to a
    chosen subset instead of everything pending in the batch.
    """
    if not extraction_ids:
        raise HTTPException(status_code=400, detail="Select at least one policy to ingest.")

    pipeline = AdaptiveIngestionPipeline()
    chosen = set(extraction_ids)

    staged = run_query(
        """
        SELECT * FROM policy_extractions_staging
        WHERE batch_id = %s AND status IN ('pending', 'blocked_schema')
        """,
        (batch_id,),
    )
    if not staged:
        raise HTTPException(status_code=404, detail="Nothing is awaiting review in this batch.")

    known = {str(row["extraction_id"]) for row in staged}
    unknown = chosen - known
    if unknown:
        raise HTTPException(status_code=400, detail="Some selected policies are not part of this batch.")

    published: list[str] = []
    failed: list[dict] = []

    for row in staged:
        extraction_id = str(row["extraction_id"])
        if extraction_id not in chosen:
            # Explicitly rejected in review; keep it out of future batches.
            pipeline.staging.mark_status(extraction_id, "skipped_by_reviewer")
            continue
        try:
            policy_id = pipeline.publisher.publish_row(row)
            pipeline.staging.mark_status(extraction_id, "published")
            published.append(str(policy_id))
        except NonPublishableChunk as e:
            pipeline.staging.mark_status(extraction_id, "skipped_non_publishable", str(e))
            failed.append({"extraction_id": extraction_id, "error": str(e)})
        except Exception as e:
            pipeline.staging.mark_status(extraction_id, "failed_publish", str(e))
            failed.append({"extraction_id": extraction_id, "error": str(e)})

    execute(
        "UPDATE ingestion_batches SET status = 'completed', finished_at = NOW() WHERE batch_id = %s",
        (batch_id,),
    )

    # Record the review decision alongside the other policy admin actions.
    for policy_id in published:
        try:
            execute(
                """
                INSERT INTO policy_admin_audit (actor_user_id, policy_id, action, details)
                VALUES (%s, %s, 'ingest_from_document', %s)
                """,
                (published_by, policy_id, json.dumps({"batch_id": batch_id})),
            )
        except Exception:
            # The audit table arrives with migration 0005; publishing must not
            # depend on it.
            logger.warning("policy_admin_audit unavailable; skipped ingest audit row")
            break

    return {
        "batch_id": batch_id,
        "published_count": len(published),
        "skipped_count": len(staged) - len(chosen),
        "failed": failed,
    }


def discard_batch(batch_id: str) -> dict[str, Any]:
    """Abandon a review without publishing anything from it."""
    changed = execute(
        """
        UPDATE policy_extractions_staging
        SET status = 'discarded_by_reviewer'
        WHERE batch_id = %s AND status IN ('pending', 'blocked_schema')
        """,
        (batch_id,),
    )
    execute(
        "UPDATE ingestion_batches SET status = 'discarded', finished_at = NOW() WHERE batch_id = %s",
        (batch_id,),
    )
    return {"batch_id": batch_id, "discarded_count": changed}
