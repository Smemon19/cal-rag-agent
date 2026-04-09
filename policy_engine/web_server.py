"""
Minimal web UI for the policy_engine pipeline (Starlette + uvicorn).

From repo root:
    python policy_engine/web_server.py

Open http://127.0.0.1:8765
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv

load_dotenv(_ROOT / ".env", override=True)

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Route

from adaptive_ingestion.pipeline import AdaptiveIngestionPipeline, IngestionDocumentInput
from policy_engine.service import answer_policy_question

_WEB_DIR = Path(__file__).resolve().parent / "web"
_PIPELINE: AdaptiveIngestionPipeline | None = None


def _get_pipeline() -> AdaptiveIngestionPipeline:
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = AdaptiveIngestionPipeline()
    return _PIPELINE


async def homepage(_request: Request) -> FileResponse:
    return FileResponse(_WEB_DIR / "index.html")


async def ask(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Expected JSON body with key \"question\"."}, status_code=400)

    question = (body.get("question") or "").strip()
    if not question:
        return JSONResponse({"error": "question is required."}, status_code=400)

    try:
        result = answer_policy_question(question)
        return JSONResponse(result)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


async def ingest_batch(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Expected JSON body."}, status_code=400)
    docs_raw = body.get("documents") or []
    if not isinstance(docs_raw, list) or not docs_raw:
        return JSONResponse({"error": "documents[] is required"}, status_code=400)
    triggered_by = str(body.get("triggered_by") or "api")
    pipeline = _get_pipeline()
    batch_id = pipeline.create_batch(triggered_by=triggered_by, trigger_source="api")
    docs = [
        IngestionDocumentInput(
            source_uri=str(d.get("source_uri") or ""),
            content=str(d.get("content") or ""),
            title=d.get("title"),
            doc_type=d.get("doc_type"),
            version=d.get("version"),
        )
        for d in docs_raw
    ]
    result = pipeline.run_batch(batch_id=batch_id, documents=docs)
    proposals = pipeline.plan_schema_changes(batch_id=batch_id)
    return JSONResponse({"batch": result, "proposals": proposals})


async def apply_migrations(_request: Request) -> JSONResponse:
    migration_ids = _get_pipeline().apply_approved_migrations(applied_by="web_server")
    return JSONResponse({"migration_ids": migration_ids})


async def publish_batch(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Expected JSON body with batch_id."}, status_code=400)
    batch_id = str(body.get("batch_id") or "").strip()
    if not batch_id:
        return JSONResponse({"error": "batch_id is required"}, status_code=400)
    result = _get_pipeline().replay_and_publish(batch_id=batch_id)
    return JSONResponse(result)


app = Starlette(
    debug=False,
    routes=[
        Route("/", homepage, methods=["GET"]),
        Route("/ask", ask, methods=["POST"]),
        Route("/ingest_batch", ingest_batch, methods=["POST"]),
        Route("/apply_migrations", apply_migrations, methods=["POST"]),
        Route("/publish_batch", publish_batch, methods=["POST"]),
    ],
)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8765, log_level="info")
