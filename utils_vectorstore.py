"""Vector Store abstraction layer for supporting multiple backends.

This module provides a unified interface for vector storage operations,
supporting both ChromaDB and BigQuery Vector Search backends.
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


class ChromaVectorStore(VectorStore):
    """ChromaDB implementation of VectorStore."""

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: Optional[str] = None,
    ):
        """Initialize ChromaDB vector store.

        Args:
            persist_directory: Directory for ChromaDB persistence
            collection_name: Name of the collection to use
        """
        from utils import (
            get_chroma_client,
            get_or_create_collection,
            get_default_chroma_dir,
            resolve_collection_name,
        )

        self.persist_directory = persist_directory or get_default_chroma_dir()
        self.collection_name = resolve_collection_name(collection_name)
        self.client = get_chroma_client(self.persist_directory)
        self.collection = get_or_create_collection(
            self.client,
            collection_name=self.collection_name
        )

    def vector_search(
        self,
        query_embedding: List[float],
        n_results: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Perform vector similarity search using ChromaDB."""
        # Convert filters to Chroma's where clause format if needed
        where = None
        if filters:
            # Chroma uses a different filter format, so we'll do post-filtering
            # for now to maintain compatibility
            pass

        # Query the collection using embeddings directly
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"]
        )

        # If we have filters, apply them post-query
        if filters:
            results = self._apply_filters(results, filters)

        return results

    def _apply_filters(
        self, results: Dict[str, Any], filters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply post-query filters to results."""
        if not filters:
            return results

        filtered_ids = []
        filtered_docs = []
        filtered_metas = []
        filtered_dists = []

        for i, (doc, meta, dist, doc_id) in enumerate(zip(
            results.get("documents", [[]])[0],
            results.get("metadatas", [[]])[0],
            results.get("distances", [[]])[0],
            results.get("ids", [[]])[0],
        )):
            keep = True

            # Apply source_contains filter
            if "source_contains" in filters and filters["source_contains"]:
                source = (
                    meta.get("source_url", "")
                    or meta.get("source", "")
                    or meta.get("file_path", "")
                )
                if filters["source_contains"].lower() not in source.lower():
                    keep = False

            # Apply header_contains filter
            if "header_contains" in filters and filters["header_contains"]:
                header = (
                    meta.get("section_path", "")
                    or meta.get("headers", "")
                    or meta.get("title", "")
                    or meta.get("header", "")
                )
                if filters["header_contains"].lower() not in header.lower():
                    keep = False

            if keep:
                filtered_ids.append(doc_id)
                filtered_docs.append(doc)
                filtered_metas.append(meta)
                filtered_dists.append(dist)

        return {
            "ids": [filtered_ids],
            "documents": [filtered_docs],
            "metadatas": [filtered_metas],
            "distances": [filtered_dists],
        }

    def keyword_search(
        self,
        terms: List[str],
        max_results: int = 5,
    ) -> Dict[str, Any]:
        """Perform keyword search using ChromaDB."""
        from utils import keyword_search_collection

        return keyword_search_collection(
            self.collection,
            substrings=terms,
            max_results=max_results,
        )

    def get_by_ids(self, ids: List[str]) -> Dict[str, Any]:
        """Fetch documents by IDs from ChromaDB."""
        if not ids:
            return {"ids": [], "documents": [], "metadatas": []}

        try:
            result = self.collection.get(
                ids=ids,
                include=["documents", "metadatas"]
            )
            return result
        except Exception as e:
            print(json.dumps({
                "where": "chroma",
                "action": "get_by_ids_error",
                "error": str(e),
            }))
            return {"ids": [], "documents": [], "metadatas": []}

    def count_documents(self) -> int:
        """Count documents in ChromaDB collection."""
        try:
            return self.collection.count()
        except Exception:
            return 0

    def get_info(self) -> Dict[str, Any]:
        """Get ChromaDB collection information."""
        try:
            count = self.count_documents()
            metadata = getattr(self.collection, "metadata", {}) or {}

            return {
                "backend": "chroma",
                "collection_name": self.collection_name,
                "persist_directory": self.persist_directory,
                "document_count": count,
                "embedding_backend": metadata.get("embedding_backend", ""),
                "embedding_model": metadata.get("embedding_model", ""),
                "distance_function": metadata.get("hnsw:space", "cosine"),
            }
        except Exception as e:
            return {
                "backend": "chroma",
                "error": str(e),
            }

    def upsert(
        self,
        ids: List[str],
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        batch_size: int = 100,
    ) -> None:
        """Insert documents into ChromaDB."""
        from utils import add_documents_to_collection

        add_documents_to_collection(
            self.collection,
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            batch_size=batch_size,
        )


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
                f"Ensure credentials are configured or use VECTOR_BACKEND=chroma"
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

        # Convert BigQuery results to Chroma-like format
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

        # Convert to Chroma-like format (with distances for consistency)
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

        # Convert to Chroma-like format
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
        backend: Vector store backend ("chroma" or "bigquery").
                 If None, reads from VECTOR_BACKEND environment variable.
        **kwargs: Additional arguments passed to the vector store constructor

    Returns:
        VectorStore instance

    Environment variables:
        VECTOR_BACKEND: "chroma" (default) or "bigquery"
    """
    # Resolve backend
    backend_name = backend or os.getenv("VECTOR_BACKEND", "chroma")
    backend_name = backend_name.strip().lower()

    print(json.dumps({
        "where": "vector_store_factory",
        "action": "create",
        "backend": backend_name,
        "ts": datetime.now(timezone.utc).isoformat(),
    }))

    if backend_name == "bigquery":
        try:
            return BigQueryVectorStore(**kwargs)
        except (RuntimeError, ValueError, Exception) as e:
            # Graceful fallback to Chroma if BigQuery fails
            print(json.dumps({
                "where": "vector_store_factory",
                "action": "bigquery_fallback_to_chroma",
                "error": str(e),
                "ts": datetime.now(timezone.utc).isoformat(),
            }))
            print(f"Warning: BigQuery initialization failed, falling back to Chroma: {e}")
            return ChromaVectorStore(**kwargs)
    elif backend_name == "chroma":
        return ChromaVectorStore(**kwargs)
    else:
        print(f"Warning: Unknown VECTOR_BACKEND='{backend_name}', falling back to chroma")
        return ChromaVectorStore(**kwargs)


def resolve_vector_backend() -> Tuple[str, Dict[str, Any]]:
    """Resolve the active vector backend and its configuration.

    Returns:
        Tuple of (backend_name, config_dict)
    """
    backend = os.getenv("VECTOR_BACKEND", "chroma").strip().lower()

    if backend == "bigquery":
        from utils_bigquery import get_bq_project, get_bq_dataset, get_bq_table
        config = {
            "backend": "bigquery",
            "project": get_bq_project(),
            "dataset": get_bq_dataset(),
            "table": get_bq_table(),
        }
    else:
        from utils import get_default_chroma_dir, get_default_collection_name
        config = {
            "backend": "chroma",
            "persist_directory": get_default_chroma_dir(),
            "collection_name": get_default_collection_name(),
        }

    return backend, config
