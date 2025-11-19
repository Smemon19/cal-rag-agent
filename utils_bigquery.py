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

    # Build table identifier
    proj = project or get_bq_project()
    ds = dataset or get_bq_dataset()
    tbl = table or get_bq_table()
    full_table = f"{proj}.{ds}.{tbl}"

    # Build WHERE clause for filters
    where_clauses = []
    query_params = []

    if filters:
        # Handle source_contains filter
        if "source_contains" in filters and filters["source_contains"]:
            where_clauses.append("LOWER(source_url) LIKE @source_pattern")
            query_params.append(
                bigquery.ScalarQueryParameter(
                    "source_pattern", "STRING", f"%{filters['source_contains'].lower()}%"
                )
            )

        # Handle header_contains filter (check metadata JSON)
        if "header_contains" in filters and filters["header_contains"]:
            where_clauses.append(
                "(LOWER(JSON_EXTRACT_SCALAR(metadata, '$.section_path')) LIKE @header_pattern "
                "OR LOWER(JSON_EXTRACT_SCALAR(metadata, '$.headers')) LIKE @header_pattern "
                "OR LOWER(JSON_EXTRACT_SCALAR(metadata, '$.title')) LIKE @header_pattern)"
            )
            query_params.append(
                bigquery.ScalarQueryParameter(
                    "header_pattern", "STRING", f"%{filters['header_contains'].lower()}%"
                )
            )

    where_clause = ""
    if where_clauses:
        where_clause = "WHERE " + " AND ".join(where_clauses)

    # Build vector search query using VECTOR_SEARCH function if index exists,
    # otherwise fall back to manual distance calculation
    # For now, we'll use manual calculation for compatibility
    query = f"""
    WITH distances AS (
        SELECT
            chunk_id,
            content,
            source_url,
            metadata,
            -- Cosine similarity using dot product (assuming normalized embeddings)
            (
                SELECT SUM(e1 * e2)
                FROM UNNEST(embedding) AS e1 WITH OFFSET pos1
                JOIN UNNEST(@query_embedding) AS e2 WITH OFFSET pos2
                ON pos1 = pos2
            ) AS similarity,
            -- Euclidean distance as fallback
            SQRT(
                (
                    SELECT SUM(POW(e1 - e2, 2))
                    FROM UNNEST(embedding) AS e1 WITH OFFSET pos1
                    JOIN UNNEST(@query_embedding) AS e2 WITH OFFSET pos2
                    ON pos1 = pos2
                )
            ) AS distance
        FROM `{full_table}`
        {where_clause}
    )
    SELECT
        chunk_id,
        content,
        source_url,
        metadata,
        similarity,
        distance,
        (1 - similarity) AS cosine_distance
    FROM distances
    ORDER BY similarity DESC
    LIMIT @limit
    """

    # Configure query parameters
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ArrayQueryParameter("query_embedding", "FLOAT64", query_embedding),
            bigquery.ScalarQueryParameter("limit", "INT64", limit),
        ] + query_params
    )

    # Execute query
    try:
        start_time = datetime.now()
        query_job = client.query(query, job_config=job_config)
        results = list(query_job.result())
        elapsed_ms = int((datetime.now() - start_time).total_seconds() * 1000)

        # Log query execution
        print(json.dumps({
            "where": "bigquery",
            "action": "vector_search",
            "results_count": len(results),
            "limit": limit,
            "elapsed_ms": elapsed_ms,
            "filters_applied": bool(filters),
        }))

        # Convert to expected format
        output = []
        for row in results:
            # Parse metadata if it's a string
            metadata = row["metadata"]
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except:
                    metadata = {}

            output.append({
                "chunk_id": row["chunk_id"],
                "content": row["content"],
                "source_url": row.get("source_url", ""),
                "metadata": metadata or {},
                "distance": float(row.get("cosine_distance", row.get("distance", 0))),
                "similarity": float(row.get("similarity", 0)),
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

    query = f"""
    SELECT
        chunk_id,
        content,
        source_url,
        metadata
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
            metadata = row["metadata"]
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except:
                    metadata = {}

            output.append({
                "chunk_id": row["chunk_id"],
                "content": row["content"],
                "source_url": row.get("source_url", ""),
                "metadata": metadata or {},
                "distance": 0.0,  # Keyword matches have no meaningful distance; set to 0 for consistency
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

    query = f"""
    SELECT
        chunk_id,
        content,
        source_url,
        metadata
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
            metadata = row["metadata"]
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except:
                    metadata = {}

            output.append({
                "chunk_id": row["chunk_id"],
                "content": row["content"],
                "source_url": row.get("source_url", ""),
                "metadata": metadata or {},
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
