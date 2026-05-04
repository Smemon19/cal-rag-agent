"""
Orchestration: plan → validate (in planner) → SQL → Postgres → optional relaxed retry → formatted answer.
The LLM planner never executes SQL; all statements are built from allowlisted columns and parameters.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from adaptive_ingestion.pipeline import AdaptiveIngestionPipeline, IngestionDocumentInput
from policy_engine.db import run_query
from policy_engine.formatter import format_answer
from policy_engine.planner import fetch_db_context, plan_search
from policy_engine.query_builder import build_query_from_spec, build_text_fallback_query, relaxed_specs_in_order
from policy_engine.synonyms import expand_question

# Cache the DB context so we don't hit the DB on every single question.
_db_context_cache: str = ""
_db_context_fetched_at: float = 0.0
_DB_CONTEXT_TTL: float = 300.0  # refresh every 5 minutes


def _get_db_context() -> str:
    global _db_context_cache, _db_context_fetched_at
    if time.time() - _db_context_fetched_at > _DB_CONTEXT_TTL:
        _db_context_cache = fetch_db_context()
        _db_context_fetched_at = time.time()
    return _db_context_cache


def _json_safe(obj: Any) -> Any:
    """Make result JSON-friendly (e.g. Decimal → str)."""
    return json.loads(json.dumps(obj, default=str))


def _extract_scope_from_question(question: str) -> dict[str, str]:
    q = question or ""
    out: dict[str, str] = {}
    m_batch = re.search(r"\b(bat_[a-z0-9]+)\b", q, flags=re.IGNORECASE)
    if m_batch:
        out["batch_id"] = m_batch.group(1)
    m_doc = re.search(r"\b(?:document_id|doc(?:ument)?)\s*[:=#]?\s*([a-z0-9_\\-]+)\b", q, flags=re.IGNORECASE)
    if m_doc:
        out["document_id"] = m_doc.group(1)
    return out


def _offline_rows_for_question(question: str, limit: int = 10) -> list[dict[str, Any]]:
    """
    Best-effort offline fallback when Postgres is temporarily unreachable.
    Uses the latest local batch report artifact if present.
    """
    candidates = [
        Path("tmp_llm_batch_report.json"),
    ]
    data_obj: dict[str, Any] | None = None
    for p in candidates:
        if not p.exists():
            continue
        try:
            data_obj = json.loads(p.read_text(encoding="utf-8"))
            break
        except Exception:
            continue
    if not data_obj:
        return []
    rows = data_obj.get("published_rows_all") or []
    if not isinstance(rows, list):
        return []

    q = (question or "").lower()
    toks = [t for t in re.findall(r"[a-z0-9_]+", q) if len(t) >= 4]
    if not toks:
        return rows[:limit]

    scored: list[tuple[int, dict[str, Any]]] = []
    for r in rows:
        blob = " ".join(
            str(r.get(k) or "")
            for k in ("summary", "topic", "subtopic", "condition_text", "action_text", "recommendation_text", "source_quote")
        ).lower()
        score = sum(1 for t in toks if t in blob)
        if score > 0:
            scored.append((score, r))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in scored[:limit]]


def answer_policy_question(question: str, *, batch_id: str | None = None, document_id: str | None = None) -> dict[str, Any]:
    """
    End-to-end policy Q&A against policies_v2.

    Flow:
    1. LLM search planner → validated search spec (JSON only, no answers).
    2. Build parameterized SELECT (LIMIT 10).
    3. Run query.
    4. If zero rows, try relaxed specs in order (drop activity_type, then category+status only).
    5. Formatter LLM produces the user-facing answer from rows only.
    """
    q = (question or "").strip()
    if not q:
        raise ValueError("Question is empty.")

    # Step 1–2: planner validates/normalizes the spec internally.
    # Ground the planner in real DB values so it doesn't guess filter values.
    expanded_question = expand_question(q)
    print(f"[Synonym Expansion] {expanded_question}")
    search_spec = plan_search(expanded_question, db_context=_get_db_context())
    scope = _extract_scope_from_question(q)
    if batch_id:
        scope["batch_id"] = batch_id
    if document_id:
        scope["document_id"] = document_id
    if scope:
        filters = dict(search_spec.get("filters") or {})
        filters.update(scope)
        search_spec["filters"] = filters

    # Step 3: primary SQL
    sql, params = build_query_from_spec(search_spec)
    try:
        rows = run_query(sql, params)
    except Exception as db_error:
        rows = _offline_rows_for_question(q)
        answer = format_answer(q, rows)
        return {
            "question": q,
            "search_spec": search_spec,
            "sql": sql,
            "params": list(params),
            "relaxed_sql": None,
            "relaxed_params": [],
            "text_fallback_sql": None,
            "text_fallback_params": [],
            "rows": _json_safe(rows),
            "row_count": len(rows),
            "answer": answer,
            "used_relaxed_query": False,
            "source": "offline_cache_fallback",
            "db_error": str(db_error),
        }

    used_relaxed = False
    relaxed_sql: str | None = None
    relaxed_params: list[Any] = []
    text_fallback_sql: str | None = None
    text_fallback_params: list[Any] = []

    # Step 4: broader retries (still parameterized SELECT, same allowlists)
    if not rows:
        for spec_r in relaxed_specs_in_order(search_spec):
            sql_r, params_r = build_query_from_spec(spec_r)
            rows_try = run_query(sql_r, params_r)
            if rows_try:
                rows = rows_try
                used_relaxed = True
                relaxed_sql = sql_r
                relaxed_params = list(params_r)
                break
    if not rows:
        tf = build_text_fallback_query(search_spec, q)
        if tf is not None:
            sql_t, params_t = tf
            rows_try = run_query(sql_t, params_t)
            if rows_try:
                rows = rows_try
                used_relaxed = True
                text_fallback_sql = sql_t
                text_fallback_params = list(params_t)

    # Step 5: answer from rows only
    answer = format_answer(q, rows)

    return {
        "question": q,
        "search_spec": search_spec,
        "sql": sql,
        "params": list(params),
        "relaxed_sql": relaxed_sql,
        "relaxed_params": relaxed_params,
        "text_fallback_sql": text_fallback_sql,
        "text_fallback_params": text_fallback_params,
        "rows": _json_safe(rows),
        "row_count": len(rows),
        "answer": answer,
        "used_relaxed_query": used_relaxed,
        "source": "structured_policy_db",
    }


def run_adaptive_ingestion_batch(
    *,
    documents: list[dict[str, Any]],
    triggered_by: str = "service",
    trigger_source: str = "service_api",
) -> dict[str, Any]:
    """Ingest documents -> stage extraction -> create schema proposals."""
    pipe = AdaptiveIngestionPipeline()
    batch_id = pipe.create_batch(triggered_by=triggered_by, trigger_source=trigger_source)
    prepared = [
        IngestionDocumentInput(
            source_uri=str(doc.get("source_uri") or ""),
            content=str(doc.get("content") or ""),
            title=doc.get("title"),
            doc_type=doc.get("doc_type"),
            version=doc.get("version"),
        )
        for doc in documents
    ]
    run_stats = pipe.run_batch(batch_id=batch_id, documents=prepared)
    proposals = pipe.plan_schema_changes(batch_id=batch_id)
    return {"batch_id": batch_id, "run_stats": run_stats, "proposals": proposals}


def apply_adaptive_migrations(applied_by: str = "service") -> list[str]:
    pipe = AdaptiveIngestionPipeline()
    return pipe.apply_approved_migrations(applied_by=applied_by)


def publish_adaptive_batch(batch_id: str) -> dict[str, Any]:
    pipe = AdaptiveIngestionPipeline()
    return pipe.replay_and_publish(batch_id=batch_id)
