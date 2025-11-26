"""Vector Store abstraction layer for supporting BigQuery Vector Search backend.

This module provides a unified interface for vector storage operations,
prioritizing BigQuery Vector Search.
"""

import os
import json
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone


class VectorStore(ABC):
    """Abstract base class for vector store implementations."""

    @abstractmethod
    def vector_search(
        self,
        query_embedding: List[float],
        n_results: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Perform vector similarity search.

        Args:
            query_embedding: Query embedding vector
            n_results: Number of results to return
            filters: Optional metadata filters

        Returns:
            Dict with keys: ids, documents, metadatas, distances
        """
        pass

    @abstractmethod
    def keyword_search(
        self,
        terms: List[str],
        max_results: int = 5,
    ) -> Dict[str, Any]:
        """Perform keyword-based search.

        Args:
            terms: List of search terms
            max_results: Maximum number of results

        Returns:
            Dict with keys: ids, documents, metadatas
        """
        pass

    @abstractmethod
    def get_by_ids(self, ids: List[str]) -> Dict[str, Any]:
        """Fetch documents by their IDs.

        Args:
            ids: List of document IDs

        Returns:
            Dict with keys: ids, documents, metadatas
        """
        pass

    @abstractmethod
    def count_documents(self) -> int:
        """Count total number of documents in the store."""
        pass

    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """Get diagnostic information about the vector store."""
        pass

    @abstractmethod
    def upsert(
        self,
        ids: List[str],
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        batch_size: int = 100,
    ) -> None:
        """Insert or update documents.

        Args:
            ids: List of document IDs
            documents: List of document texts
            metadatas: Optional list of metadata dictionaries
            batch_size: Batch size for insertion
        """
        pass


class BigQueryVectorStore(VectorStore):
    """BigQuery Vector Search implementation of VectorStore."""

    def __init__(
        self,
        project: Optional[str] = None,
        dataset: Optional[str] = None,
        table: Optional[str] = None,
    ):
        """Initialize BigQuery vector store.

        Args:
            project: BigQuery project ID
            dataset: BigQuery dataset ID
            table: BigQuery table name

        Raises:
            RuntimeError: If BigQuery client cannot be initialized (e.g., missing credentials)
        """
        from utils_bigquery import get_bq_project, get_bq_dataset, get_bq_table, get_bq_client

        self.project = project or get_bq_project()
        self.dataset = dataset or get_bq_dataset()
        self.table = table or get_bq_table()

        # Validate BigQuery connection early
        try:
            # Try to create a client to validate credentials
            _ = get_bq_client(project=self.project)
        except Exception as e:
            creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            error_msg = (
                f"Failed to initialize BigQuery client: {e}\n"
                f"GOOGLE_APPLICATION_CREDENTIALS: {creds_path or '(not set)'}\n"
                f"Ensure credentials are configured."
            )
            print(json.dumps({
                "where": "bigquery_vector_store",
                "action": "init_error",
                "error": str(e),
                "creds_path": creds_path,
            }))
            raise RuntimeError(error_msg) from e

    def vector_search(
        self,
        query_embedding: List[float],
        n_results: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Perform vector search using BigQuery."""
        from utils_bigquery import vector_search as bq_vector_search

        results = bq_vector_search(
            query_embedding=query_embedding,
            limit=n_results,
            filters=filters,
            project=self.project,
            dataset=self.dataset,
            table=self.table,
        )

        # Convert BigQuery results to standard format
        ids = [r["chunk_id"] for r in results]
        documents = [r["content"] for r in results]
        metadatas = [r["metadata"] for r in results]
        distances = [r["distance"] for r in results]

        return {
            "ids": [ids],
            "documents": [documents],
            "metadatas": [metadatas],
            "distances": [distances],
        }

    def keyword_search(
        self,
        terms: List[str],
        max_results: int = 5,
    ) -> Dict[str, Any]:
        """Perform keyword search using BigQuery."""
        from utils_bigquery import keyword_search as bq_keyword_search

        results = bq_keyword_search(
            terms=terms,
            limit=max_results,
            project=self.project,
            dataset=self.dataset,
            table=self.table,
        )

        # Convert to standard format (with distances for consistency)
        ids = [r["chunk_id"] for r in results]
        documents = [r["content"] for r in results]
        metadatas = [r["metadata"] for r in results]
        distances = [r.get("distance", 0.0) for r in results]

        return {
            "ids": [ids],
            "documents": [documents],
            "metadatas": [metadatas],
            "distances": [distances],
        }

    def get_by_ids(self, ids: List[str]) -> Dict[str, Any]:
        """Fetch documents by IDs from BigQuery."""
        from utils_bigquery import fetch_chunks_by_ids

        results = fetch_chunks_by_ids(
            chunk_ids=ids,
            project=self.project,
            dataset=self.dataset,
            table=self.table,
        )

        # Convert to standard format
        result_ids = [r["chunk_id"] for r in results]
        documents = [r["content"] for r in results]
        metadatas = [r["metadata"] for r in results]

        return {
            "ids": result_ids,
            "documents": documents,
            "metadatas": metadatas,
        }

    def count_documents(self) -> int:
        """Count documents in BigQuery table."""
        from utils_bigquery import count_documents

        return count_documents(
            project=self.project,
            dataset=self.dataset,
            table=self.table,
        )

    def get_info(self) -> Dict[str, Any]:
        """Get BigQuery table information."""
        from utils_bigquery import get_table_info, get_vector_index_info

        table_info = get_table_info(
            project=self.project,
            dataset=self.dataset,
            table=self.table,
        )

        index_info = get_vector_index_info(
            project=self.project,
            dataset=self.dataset,
            table=self.table,
        )

        return {
            "backend": "bigquery",
            "project": self.project,
            "dataset": self.dataset,
            "table": self.table,
            "document_count": table_info.get("num_rows", 0),
            "table_size_bytes": table_info.get("num_bytes", 0),
            "created": table_info.get("created"),
            "modified": table_info.get("modified"),
            "indexes": index_info,
        }

    def upsert(
        self,
        ids: List[str],
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        batch_size: int = 100,
    ) -> None:
        """Insert documents into BigQuery.

        Note: This is a placeholder. BigQuery insertions are typically
        done via batch ingestion scripts rather than real-time upserts.
        """
        raise NotImplementedError(
            "BigQuery insertions should be done via batch ingestion scripts. "
            "See scripts/ingest_to_bigquery.py for the canonical ingestion path."
        )


def get_vector_store(
    backend: Optional[str] = None,
    **kwargs,
) -> VectorStore:
    """Factory function to create appropriate vector store instance.

    Args:
        backend: Vector store backend ("bigquery").
                 If None, defaults to "bigquery".
        **kwargs: Additional arguments passed to the vector store constructor

    Returns:
        VectorStore instance

    Environment variables:
        VECTOR_BACKEND: "bigquery" (default)
    """
    # Resolve backend - default to bigquery
    backend_name = backend or os.getenv("VECTOR_BACKEND", "bigquery")
    backend_name = backend_name.strip().lower()

    # Warn if someone explicitly requested chroma
    if backend_name == "chroma":
        print("Warning: VECTOR_BACKEND='chroma' is deprecated. Defaulting to 'bigquery'.")
        backend_name = "bigquery"

    print(json.dumps({
        "where": "vector_store_factory",
        "action": "create",
        "backend": backend_name,
        "ts": datetime.now(timezone.utc).isoformat(),
    }))

    return BigQueryVectorStore(**kwargs)


def resolve_vector_backend() -> Tuple[str, Dict[str, Any]]:
    """Resolve the active vector backend and its configuration.

    Returns:
        Tuple of (backend_name, config_dict)
    """
    backend = "bigquery"
    
    from utils_bigquery import get_bq_project, get_bq_dataset, get_bq_table
    config = {
        "backend": "bigquery",
        "project": get_bq_project(),
        "dataset": get_bq_dataset(),
        "table": get_bq_table(),
    }

    return backend, config
