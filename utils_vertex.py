"""Vertex AI embedding function using google-auth + REST API.

Calls the Vertex AI predict endpoint for text embeddings without requiring
the heavy google-cloud-aiplatform SDK.  Authentication uses Application
Default Credentials (ADC), which works transparently on Cloud Run with an
attached service account and locally after `gcloud auth application-default login`.
"""

from __future__ import annotations

import json
import os
import time
import random
from typing import List, Optional

import httpx

try:
    import google.auth
    import google.auth.transport.requests
    from google.auth.credentials import Credentials as BaseCredentials
except ImportError as _exc:
    raise ImportError(
        "google-auth is required for Vertex AI embeddings. "
        "Install it with: pip install google-auth"
    ) from _exc


# Vertex AI predict endpoint template
_VERTEX_PREDICT_URL = (
    "https://{location}-aiplatform.googleapis.com/v1"
    "/projects/{project_id}"
    "/locations/{location}"
    "/publishers/google"
    "/models/{model}:predict"
)

# Default output dimension for known models
_KNOWN_DIMS = {
    "text-embedding-005": 768,
    "text-embedding-004": 768,
    "text-multilingual-embedding-002": 768,
    "gemini-embedding-001": 768,  # default; supports up to 3072
}

# Default batch size — conservative to avoid quota / payload issues.
# Vertex API hard limit is 250, but 32 is safer for typical workloads.
# Override via VERTEX_EMBED_BATCH_SIZE env var.
_DEFAULT_BATCH_SIZE = 32
_MAX_BATCH_SIZE = 250


def _get_vertex_project() -> str:
    """Resolve Vertex project ID from environment with fallbacks."""
    for name in ("VERTEX_PROJECT_ID", "BQ_PROJECT", "BQ_PROJECT_ID", "GOOGLE_CLOUD_PROJECT"):
        val = (os.getenv(name) or "").strip()
        if val:
            return val
    raise RuntimeError(
        "Cannot determine Vertex AI project ID. "
        "Set VERTEX_PROJECT_ID, BQ_PROJECT, or GOOGLE_CLOUD_PROJECT."
    )


def _get_vertex_location() -> str:
    # Assured Workloads deployment default (overridable via env).
    return (os.getenv("VERTEX_LOCATION") or "").strip() or "us-central1"


def _get_vertex_embedding_model() -> str:
    return (os.getenv("VERTEX_EMBEDDING_MODEL") or "").strip() or "text-embedding-005"


class VertexEmbeddingFunction:
    """Embedding callable that uses Vertex AI text-embedding models via REST.

    Compatible with the embedding function interface used by the rest of
    the codebase: ``fn(texts: List[str]) -> List[List[float]]``.
    """

    def __init__(
        self,
        project_id: Optional[str] = None,
        location: Optional[str] = None,
        model_name: Optional[str] = None,
        task_type: str = "RETRIEVAL_DOCUMENT",
        batch_size: Optional[int] = None,
    ):
        self._project_id = project_id or _get_vertex_project()
        self._location = location or _get_vertex_location()
        self._model_name = model_name or _get_vertex_embedding_model()
        self._task_type = task_type

        # Resolve batch size: explicit arg > env var > default (32)
        if batch_size is None:
            try:
                batch_size = int(os.getenv("VERTEX_EMBED_BATCH_SIZE", str(_DEFAULT_BATCH_SIZE)))
            except (ValueError, TypeError):
                batch_size = _DEFAULT_BATCH_SIZE
        self._batch_size = max(1, min(batch_size, _MAX_BATCH_SIZE))

        self._url = _VERTEX_PREDICT_URL.format(
            location=self._location,
            project_id=self._project_id,
            model=self._model_name,
        )

        # Resolve expected dimension (used for zero-vector fallback).
        # _dim_validated flips to True after the first real API response,
        # at which point we enforce consistency for all subsequent batches.
        self._dim = _KNOWN_DIMS.get(self._model_name, 768)
        self._dim_validated = False

        # Auth: use ADC (works on Cloud Run and locally with gcloud CLI)
        self._credentials: BaseCredentials
        self._credentials, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        self._auth_request = google.auth.transport.requests.Request()

        # Synchronous httpx client for embedding calls
        self._client = httpx.Client(timeout=60.0)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def name(self) -> str:
        return f"vertex::{self._model_name}"

    @property
    def dim(self) -> int:
        return self._dim

    def __call__(self, input: List[str]) -> List[List[float]]:  # noqa: A002
        """Embed a list of texts and return a list of float vectors."""
        if not input:
            return []

        all_embeddings: List[List[float]] = [[] for _ in range(len(input))]

        for batch_start in range(0, len(input), self._batch_size):
            batch_end = min(batch_start + self._batch_size, len(input))
            batch_texts = input[batch_start:batch_end]

            # Build instances; skip empty strings (will get zero vectors)
            instances = []
            real_indices = []  # maps instance position -> original index
            for i, text in enumerate(batch_texts):
                if text and text.strip():
                    instances.append({"content": text, "task_type": self._task_type})
                    real_indices.append(batch_start + i)

            if not instances:
                # All texts in this batch are empty
                for i in range(batch_start, batch_end):
                    all_embeddings[i] = [0.0] * self._dim
                continue

            embeddings = self._call_predict(instances)

            # Place results at correct indices
            emb_idx = 0
            for i in range(batch_start, batch_end):
                if i in real_indices:
                    if emb_idx < len(embeddings):
                        all_embeddings[i] = embeddings[emb_idx]
                        emb_idx += 1
                    else:
                        all_embeddings[i] = [0.0] * self._dim
                else:
                    all_embeddings[i] = [0.0] * self._dim

        # Final safety: verify every vector has the expected dimension
        if self._dim_validated:
            for idx, emb in enumerate(all_embeddings):
                if len(emb) != self._dim:
                    raise RuntimeError(
                        f"Embedding at index {idx} has {len(emb)} dims, "
                        f"expected {self._dim}. This indicates an internal bug "
                        f"or a model dimension change."
                    )

        return all_embeddings

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _refresh_token(self) -> str:
        """Refresh credentials and return a valid bearer token."""
        if not self._credentials.valid:
            self._credentials.refresh(self._auth_request)
        return self._credentials.token  # type: ignore[return-value]

    def _call_predict(self, instances: List[dict], _attempt: int = 0) -> List[List[float]]:
        """Call the Vertex AI predict endpoint with retry on transient errors."""
        token = self._refresh_token()
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        body = {"instances": instances}

        try:
            resp = self._client.post(self._url, headers=headers, json=body)
        except httpx.HTTPError as exc:
            if _attempt < 1:
                time.sleep(0.5 + random.random())
                return self._call_predict(instances, _attempt + 1)
            raise RuntimeError(f"Vertex AI embedding request failed: {exc}") from exc

        if resp.status_code in (429, 500, 502, 503, 504) and _attempt < 1:
            wait = 1.0 + random.random()
            print(
                json.dumps({
                    "where": "vertex_embeddings",
                    "action": "retry",
                    "status": resp.status_code,
                    "wait_s": round(wait, 2),
                })
            )
            time.sleep(wait)
            return self._call_predict(instances, _attempt + 1)

        if resp.status_code != 200:
            raise RuntimeError(
                f"Vertex AI embedding error {resp.status_code}: {resp.text[:500]}"
            )

        data = resp.json()
        predictions = data.get("predictions", [])

        embeddings: List[List[float]] = []
        for pred in predictions:
            emb = pred.get("embeddings", {})
            values = emb.get("values", [])
            if values:
                actual_len = len(values)
                # On first real response, learn the true dimension
                if not self._dim_validated:
                    self._dim = actual_len
                    self._dim_validated = True
                elif actual_len != self._dim:
                    raise RuntimeError(
                        f"Vertex AI returned embedding with {actual_len} dims, "
                        f"expected {self._dim}. Model may have changed or "
                        f"outputDimensionality is misconfigured."
                    )
                embeddings.append(values)
            else:
                embeddings.append([0.0] * self._dim)

        return embeddings
