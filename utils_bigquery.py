"""BigQuery Vector Store utilities for RAG operations.

This module provides BigQuery-specific operations for vector search and document retrieval.
It is designed to replace ChromaDB operations with BigQuery Vector Search.
"""

import os
import json
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from google.cloud import bigquery
from google.cloud.bigquery import Client


# Cache for BigQuery client
_BQ_CLIENT_CACHE: Optional[Client] = None
_BQ_LAST_STATUS: Dict[str, Any] = {}
_BQ_EMBED_CONFIG_CACHE: Dict[tuple[str, str, str], tuple[Optional[str], Optional[str]]] = {}


def get_bq_project() -> str:
    """Get BigQuery project ID from environment."""
    return os.getenv("BQ_PROJECT", "cal-rag-agent")


def get_bq_dataset() -> str:
    """Get BigQuery dataset ID from environment."""
    return os.getenv("BQ_DATASET", "calrag")


def get_bq_table() -> str:
    """Get BigQuery table name from environment."""
    return os.getenv("BQ_TABLE", "documents")


def get_full_table_id() -> str:
    """Get fully qualified BigQuery table ID."""
    return f"{get_bq_project()}.{get_bq_dataset()}.{get_bq_table()}"


def get_table_embedding_config(
    project: Optional[str] = None,
    dataset: Optional[str] = None,
    table: Optional[str] = None,
) -> tuple[Optional[str], Optional[str]]:
    """Return (embedding_backend, embedding_model) metadata stored with documents.

    Results are cached per table to avoid repeated metadata queries.
    """
    global _BQ_EMBED_CONFIG_CACHE

    proj = project or get_bq_project()
    ds = dataset or get_bq_dataset()
    tbl = table or get_bq_table()
    cache_key = (proj, ds, tbl)

    if cache_key in _BQ_EMBED_CONFIG_CACHE:
        return _BQ_EMBED_CONFIG_CACHE[cache_key]

    client = get_bq_client(proj)
    full_table = f"{proj}.{ds}.{tbl}"

    query = f"""
    SELECT
        embedding_backend,
        embedding_model
    FROM `{full_table}`
    WHERE embedding_backend IS NOT NULL
    LIMIT 1
    """

    try:
        results = list(client.query(query).result())
        if not results:
            _BQ_EMBED_CONFIG_CACHE[cache_key] = (None, None)
            return (None, None)

        row = results[0]
        backend = str(row.get("embedding_backend") or "").strip() or None
        model = str(row.get("embedding_model") or "").strip() or None
        _BQ_EMBED_CONFIG_CACHE[cache_key] = (backend, model)
        return backend, model
    except Exception as e:
        print(json.dumps({
            "where": "bigquery",
            "action": "embedding_config_error",
            "error": str(e),
            "table": full_table,
        }))
        _BQ_EMBED_CONFIG_CACHE[cache_key] = (None, None)
        return (None, None)


def get_bq_client(project: Optional[str] = None) -> Client:
    """Get or create a BigQuery client instance.

    Args:
        project: Optional project ID. If None, uses BQ_PROJECT environment variable.

    Returns:
        BigQuery Client instance

    Raises:
        ValueError: If GOOGLE_APPLICATION_CREDENTIALS is set but file doesn't exist
        Exception: If BigQuery client initialization fails
    """
    global _BQ_CLIENT_CACHE, _BQ_LAST_STATUS

    project_id = project or get_bq_project()

    # Validate GOOGLE_APPLICATION_CREDENTIALS if set
    creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if creds_path:
        from pathlib import Path
        creds_file = Path(creds_path)
        if not creds_file.exists():
            error_msg = f"GOOGLE_APPLICATION_CREDENTIALS points to non-existent file: {creds_path}"
            error_status = {
                "where": "bigquery",
                "action": "init_error",
                "error": error_msg,
                "project": project_id,
                "creds_path": creds_path,
                "ts": datetime.now(timezone.utc).isoformat(),
            }
            _BQ_LAST_STATUS = error_status
            print(json.dumps(error_status))
            raise ValueError(error_msg)

    # Return cached client if available and project matches
    if _BQ_CLIENT_CACHE is not None:
        if _BQ_CLIENT_CACHE.project == project_id:
            status = {
                "where": "bigquery",
                "action": "init_or_reuse",
                "target": "client",
                "project": project_id,
                "reused": True,
                "creds_configured": bool(creds_path),
                "ts": datetime.now(timezone.utc).isoformat(),
            }
            _BQ_LAST_STATUS = status
            print(json.dumps(status))
            return _BQ_CLIENT_CACHE

    # Create new client
    try:
        client = bigquery.Client(project=project_id)
        _BQ_CLIENT_CACHE = client
        status = {
            "where": "bigquery",
            "action": "init_or_reuse",
            "target": "client",
            "project": project_id,
            "reused": False,
            "creds_configured": bool(creds_path),
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        _BQ_LAST_STATUS = status
        print(json.dumps(status))
        return client
    except Exception as e:
        error_msg = str(e)
        # Provide helpful error messages for common authentication issues
        if "could not automatically determine credentials" in error_msg.lower():
            error_msg += " | Hint: Set GOOGLE_APPLICATION_CREDENTIALS to your service account key path"
        elif "permission" in error_msg.lower() or "forbidden" in error_msg.lower():
            error_msg += " | Hint: Check service account has BigQuery permissions"

        error_status = {
            "where": "bigquery",
            "action": "init_error",
            "error": error_msg,
            "project": project_id,
            "creds_configured": bool(creds_path),
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        _BQ_LAST_STATUS = error_status
        print(json.dumps(error_status))
        raise


def vector_search(
    query_embedding: List[float],
    limit: int = 5,
    filters: Optional[Dict[str, Any]] = None,
    project: Optional[str] = None,
    dataset: Optional[str] = None,
    table: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Perform vector similarity search using BigQuery Vector Search.

    Args:
        query_embedding: Query embedding vector
        limit: Maximum number of results to return
        filters: Optional metadata filters (e.g., {"source_contains": "example.com"})
        project: Optional project override
        dataset: Optional dataset override
        table: Optional table override

    Returns:
        List of result dictionaries with keys: chunk_id, content, metadata, distance
    """
    client = get_bq_client(project)

    proj = project or get_bq_project()
    ds = dataset or get_bq_dataset()
    tbl = table or get_bq_table()
    full_table = f"{proj}.{ds}.{tbl}"

    # Allow fetching extra candidates so post-filters still return enough rows
    candidate_limit = max(limit, min(limit * 5, 200))

    where_clauses: List[str] = []
    query_params: List[bigquery.ScalarQueryParameter] = []

    if filters:
        src = filters.get("source_contains")
        if src:
            where_clauses.append("LOWER(vs.base.source_url) LIKE @source_pattern")
            query_params.append(
                bigquery.ScalarQueryParameter("source_pattern", "STRING", f"%{src.lower()}%")
            )
        hdr = filters.get("header_contains")
        if hdr:
            where_clauses.append(
                "("
                "LOWER(vs.base.section_path) LIKE @header_pattern OR "
                "LOWER(vs.base.headers) LIKE @header_pattern OR "
                "LOWER(vs.base.title) LIKE @header_pattern"
                ")"
            )
            query_params.append(
                bigquery.ScalarQueryParameter("header_pattern", "STRING", f"%{hdr.lower()}%")
            )

    where_clause = ""
    if where_clauses:
        where_clause = "WHERE " + " AND ".join(where_clauses)

    query = f"""
    SELECT
        vs.base.chunk_id AS chunk_id,
        vs.base.content AS content,
        vs.base.source_url AS source_url,
        vs.base.source_type AS source_type,
        vs.base.section_path AS section_path,
        vs.base.headers AS headers,
        vs.base.page_number AS page_number,
        vs.base.title AS title,
        vs.base.mime_type AS mime_type,
        vs.base.char_count AS char_count,
        vs.base.word_count AS word_count,
        vs.base.content_preview AS content_preview,
        vs.base.embedding_backend AS embedding_backend,
        vs.base.embedding_model AS embedding_model,
        vs.base.inserted_at AS inserted_at,
        vs.distance AS distance,
        1 - vs.distance AS similarity
    FROM VECTOR_SEARCH(
        TABLE `{full_table}`,
        'embedding',
        (SELECT @query_embedding),
        top_k => @candidate_limit
    ) AS vs
    {where_clause}
    ORDER BY distance ASC
    LIMIT @limit
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ArrayQueryParameter("query_embedding", "FLOAT64", query_embedding),
            bigquery.ScalarQueryParameter("candidate_limit", "INT64", candidate_limit),
            bigquery.ScalarQueryParameter("limit", "INT64", limit),
        ] + query_params
    )

    try:
        start_time = datetime.now()
        query_job = client.query(query, job_config=job_config)
        results = list(query_job.result())
        elapsed_ms = int((datetime.now() - start_time).total_seconds() * 1000)

        print(json.dumps({
            "where": "bigquery",
            "action": "vector_search",
            "results_count": len(results),
            "limit": limit,
            "candidate_limit": candidate_limit,
            "elapsed_ms": elapsed_ms,
            "filters_applied": bool(filters),
        }))

        output = []
        for row in results:
            metadata = {
                "source_url": row.get("source_url"),
                "source_type": row.get("source_type"),
                "section_path": row.get("section_path"),
                "headers": row.get("headers"),
                "page_number": row.get("page_number"),
                "title": row.get("title"),
                "mime_type": row.get("mime_type"),
                "char_count": row.get("char_count"),
                "word_count": row.get("word_count"),
                "content_preview": row.get("content_preview"),
                "embedding_backend": row.get("embedding_backend"),
                "embedding_model": row.get("embedding_model"),
                "inserted_at": row.get("inserted_at").isoformat() if row.get("inserted_at") else None,
            }
            output.append({
                "chunk_id": row.get("chunk_id"),
                "content": row.get("content"),
                "source_url": row.get("source_url") or "",
                "metadata": metadata,
                "distance": float(row.get("distance") or 0.0),
                "similarity": float(row.get("similarity") or 0.0),
            })

        return output

    except Exception as e:
        print(json.dumps({
            "where": "bigquery",
            "action": "vector_search_error",
            "error": str(e),
            "limit": limit,
        }))
        raise


def keyword_search(
    terms: List[str],
    limit: int = 5,
    project: Optional[str] = None,
    dataset: Optional[str] = None,
    table: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Perform keyword-based search using SQL LIKE queries.

    This is a fallback for exact literal/section matching (e.g., "TABLE 1507.9.6").

    Args:
        terms: List of keywords/phrases to search for
        limit: Maximum number of results
        project: Optional project override
        dataset: Optional dataset override
        table: Optional table override

    Returns:
        List of result dictionaries with keys: chunk_id, content, metadata
    """
    if not terms:
        return []

    client = get_bq_client(project)

    # Build table identifier
    proj = project or get_bq_project()
    ds = dataset or get_bq_dataset()
    tbl = table or get_bq_table()
    full_table = f"{proj}.{ds}.{tbl}"

    # Build OR conditions for each term
    conditions = []
    query_params = []

    for i, term in enumerate(terms):
        if term and term.strip():
            param_name = f"term_{i}"
            conditions.append(f"LOWER(content) LIKE @{param_name}")
            query_params.append(
                bigquery.ScalarQueryParameter(
                    param_name, "STRING", f"%{term.lower()}%"
                )
            )

    if not conditions:
        return []

    where_clause = " OR ".join(conditions)

    # Select specific columns instead of 'metadata'
    query = f"""
    SELECT
        chunk_id,
        content,
        source_url,
        source_type,
        section_path,
        headers,
        page_number,
        title,
        mime_type,
        char_count,
        word_count,
        content_preview,
        embedding_backend,
        embedding_model,
        inserted_at
    FROM `{full_table}`
    WHERE {where_clause}
    LIMIT @limit
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=query_params + [
            bigquery.ScalarQueryParameter("limit", "INT64", limit),
        ]
    )

    try:
        start_time = datetime.now()
        query_job = client.query(query, job_config=job_config)
        results = list(query_job.result())
        elapsed_ms = int((datetime.now() - start_time).total_seconds() * 1000)

        print(json.dumps({
            "where": "bigquery",
            "action": "keyword_search",
            "terms_count": len(terms),
            "results_count": len(results),
            "elapsed_ms": elapsed_ms,
        }))

        # Convert to expected format
        output = []
        for row in results:
            # Reconstruct metadata
            metadata = {
                "source_url": row.get("source_url"),
                "source_type": row.get("source_type"),
                "section_path": row.get("section_path"),
                "headers": row.get("headers"),
                "page_number": row.get("page_number"),
                "title": row.get("title"),
                "mime_type": row.get("mime_type"),
                "char_count": row.get("char_count"),
                "word_count": row.get("word_count"),
                "content_preview": row.get("content_preview"),
                "embedding_backend": row.get("embedding_backend"),
                "embedding_model": row.get("embedding_model"),
                "inserted_at": row.get("inserted_at").isoformat() if row.get("inserted_at") else None,
            }

            output.append({
                "chunk_id": row["chunk_id"],
                "content": row["content"],
                "source_url": row.get("source_url", ""),
                "metadata": metadata,
                "distance": 0.0,
            })

        return output

    except Exception as e:
        print(json.dumps({
            "where": "bigquery",
            "action": "keyword_search_error",
            "error": str(e),
        }))
        raise


def fetch_chunks_by_ids(
    chunk_ids: List[str],
    project: Optional[str] = None,
    dataset: Optional[str] = None,
    table: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Fetch specific chunks by their IDs.

    Args:
        chunk_ids: List of chunk IDs to fetch
        project: Optional project override
        dataset: Optional dataset override
        table: Optional table override

    Returns:
        List of chunk dictionaries
    """
    if not chunk_ids:
        return []

    client = get_bq_client(project)

    proj = project or get_bq_project()
    ds = dataset or get_bq_dataset()
    tbl = table or get_bq_table()
    full_table = f"{proj}.{ds}.{tbl}"

    # Select specific columns
    query = f"""
    SELECT
        chunk_id,
        content,
        source_url,
        source_type,
        section_path,
        headers,
        page_number,
        title,
        mime_type,
        char_count,
        word_count,
        content_preview,
        embedding_backend,
        embedding_model,
        inserted_at
    FROM `{full_table}`
    WHERE chunk_id IN UNNEST(@chunk_ids)
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ArrayQueryParameter("chunk_ids", "STRING", chunk_ids),
        ]
    )

    try:
        query_job = client.query(query, job_config=job_config)
        results = list(query_job.result())

        output = []
        for row in results:
            # Reconstruct metadata
            metadata = {
                "source_url": row.get("source_url"),
                "source_type": row.get("source_type"),
                "section_path": row.get("section_path"),
                "headers": row.get("headers"),
                "page_number": row.get("page_number"),
                "title": row.get("title"),
                "mime_type": row.get("mime_type"),
                "char_count": row.get("char_count"),
                "word_count": row.get("word_count"),
                "content_preview": row.get("content_preview"),
                "embedding_backend": row.get("embedding_backend"),
                "embedding_model": row.get("embedding_model"),
                "inserted_at": row.get("inserted_at").isoformat() if row.get("inserted_at") else None,
            }

            output.append({
                "chunk_id": row["chunk_id"],
                "content": row["content"],
                "source_url": row.get("source_url", ""),
                "metadata": metadata,
            })

        return output

    except Exception as e:
        print(json.dumps({
            "where": "bigquery",
            "action": "fetch_by_ids_error",
            "error": str(e),
        }))
        raise


def count_documents(
    project: Optional[str] = None,
    dataset: Optional[str] = None,
    table: Optional[str] = None,
) -> int:
    """Count total number of documents in the BigQuery table.

    Args:
        project: Optional project override
        dataset: Optional dataset override
        table: Optional table override

    Returns:
        Total row count
    """
    client = get_bq_client(project)

    proj = project or get_bq_project()
    ds = dataset or get_bq_dataset()
    tbl = table or get_bq_table()
    full_table = f"{proj}.{ds}.{tbl}"

    query = f"SELECT COUNT(*) as count FROM `{full_table}`"

    try:
        query_job = client.query(query)
        results = list(query_job.result())
        if results:
            return int(results[0]["count"])
        return 0
    except Exception as e:
        print(json.dumps({
            "where": "bigquery",
            "action": "count_error",
            "error": str(e),
        }))
        return 0


def get_table_info(
    project: Optional[str] = None,
    dataset: Optional[str] = None,
    table: Optional[str] = None,
) -> Dict[str, Any]:
    """Get metadata about the BigQuery table.

    Args:
        project: Optional project override
        dataset: Optional dataset override
        table: Optional table override

    Returns:
        Dictionary with table metadata
    """
    client = get_bq_client(project)

    proj = project or get_bq_project()
    ds = dataset or get_bq_dataset()
    tbl = table or get_bq_table()
    full_table = f"{proj}.{ds}.{tbl}"

    try:
        table_ref = client.get_table(full_table)

        return {
            "project": proj,
            "dataset": ds,
            "table": tbl,
            "full_table_id": full_table,
            "num_rows": table_ref.num_rows or 0,
            "num_bytes": table_ref.num_bytes or 0,
            "created": table_ref.created.isoformat() if table_ref.created else None,
            "modified": table_ref.modified.isoformat() if table_ref.modified else None,
            "schema_fields": len(table_ref.schema),
        }
    except Exception as e:
        print(json.dumps({
            "where": "bigquery",
            "action": "get_table_info_error",
            "error": str(e),
        }))
        return {
            "project": proj,
            "dataset": ds,
            "table": tbl,
            "error": str(e),
        }


def get_vector_index_info(
    project: Optional[str] = None,
    dataset: Optional[str] = None,
    table: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Get information about vector indexes on the table.

    Args:
        project: Optional project override
        dataset: Optional dataset override
        table: Optional table override

    Returns:
        List of index information dictionaries
    """
    client = get_bq_client(project)

    proj = project or get_bq_project()
    ds = dataset or get_bq_dataset()
    tbl = table or get_bq_table()

    query = f"""
    SELECT
        index_name,
        table_name,
        index_status,
        coverage_percentage,
        ddl
    FROM `{proj}.{ds}.INFORMATION_SCHEMA.VECTOR_INDEXES`
    WHERE table_name = @table_name
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("table_name", "STRING", tbl),
        ]
    )

    try:
        query_job = client.query(query, job_config=job_config)
        results = list(query_job.result())

        indexes = []
        for row in results:
            indexes.append({
                "index_name": row.get("index_name", ""),
                "table_name": row.get("table_name", ""),
                "status": row.get("index_status", ""),
                "coverage_percentage": row.get("coverage_percentage"),
                "ddl": row.get("ddl", ""),
            })

        return indexes
    except Exception as e:
        print(json.dumps({
            "where": "bigquery",
            "action": "get_index_info_error",
            "error": str(e),
        }))
        return []


def get_last_status() -> Dict[str, Any]:
    """Get the last BigQuery operation status for diagnostics."""
    return _BQ_LAST_STATUS.copy() if _BQ_LAST_STATUS else {}
