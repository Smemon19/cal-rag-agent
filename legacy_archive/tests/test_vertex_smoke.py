"""Smoke tests for Vertex AI integration.

These tests require valid Google Cloud ADC credentials:
    gcloud auth application-default login

Run with:
    pytest tests/test_vertex_smoke.py -v
"""

import os
import sys
import pytest

# Ensure project root is on the path so imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _skip_if_no_adc():
    """Skip the test if Google ADC is not configured."""
    try:
        import google.auth
        creds, _ = google.auth.default()
        if creds is None:
            pytest.skip("No ADC credentials available")
    except Exception as exc:
        pytest.skip(f"ADC not configured: {exc}")


# ---------------------------------------------------------------------------
# 1. Vertex Embedding Function
# ---------------------------------------------------------------------------

class TestVertexEmbeddingFunction:
    """Tests for VertexEmbeddingFunction (requires live Vertex AI access)."""

    def test_single_text_returns_768_dims(self):
        _skip_if_no_adc()
        from utils_vertex import VertexEmbeddingFunction

        fn = VertexEmbeddingFunction()
        result = fn(["What is the wind speed requirement for ASCE 7?"])
        assert len(result) == 1, "Expected exactly one embedding"
        assert len(result[0]) == 768, f"Expected 768 dims, got {len(result[0])}"
        assert all(isinstance(v, float) for v in result[0])

    def test_batch_embedding(self):
        _skip_if_no_adc()
        from utils_vertex import VertexEmbeddingFunction

        fn = VertexEmbeddingFunction()
        texts = [f"Test sentence number {i}" for i in range(5)]
        result = fn(texts)
        assert len(result) == 5
        for emb in result:
            assert len(emb) == 768

    def test_empty_string_returns_zero_vector(self):
        _skip_if_no_adc()
        from utils_vertex import VertexEmbeddingFunction

        fn = VertexEmbeddingFunction()
        result = fn(["", "hello world", ""])
        assert len(result) == 3
        # Empty strings get zero vectors
        assert result[0] == [0.0] * 768
        assert result[2] == [0.0] * 768
        # Non-empty string gets a real embedding
        assert any(v != 0.0 for v in result[1])

    def test_empty_list(self):
        _skip_if_no_adc()
        from utils_vertex import VertexEmbeddingFunction

        fn = VertexEmbeddingFunction()
        result = fn([])
        assert result == []

    def test_name_method(self):
        _skip_if_no_adc()
        from utils_vertex import VertexEmbeddingFunction

        fn = VertexEmbeddingFunction()
        assert fn.name().startswith("vertex::")

    def test_dimension_validated_after_first_call(self):
        _skip_if_no_adc()
        from utils_vertex import VertexEmbeddingFunction

        fn = VertexEmbeddingFunction()
        assert fn._dim_validated is False
        fn(["test"])
        assert fn._dim_validated is True
        assert fn.dim == 768

    def test_batch_size_from_env(self):
        """Verify VERTEX_EMBED_BATCH_SIZE env var is respected."""
        os.environ["VERTEX_EMBED_BATCH_SIZE"] = "16"
        try:
            from utils_vertex import VertexEmbeddingFunction
            fn = VertexEmbeddingFunction()
            assert fn._batch_size == 16
        finally:
            del os.environ["VERTEX_EMBED_BATCH_SIZE"]


# ---------------------------------------------------------------------------
# 2. LLM (pydantic-ai VertexAIModel)
# ---------------------------------------------------------------------------

class TestVertexLLM:
    """Tests for Gemini LLM via pydantic-ai VertexAIModel (requires live access)."""

    def test_basic_completion(self):
        _skip_if_no_adc()
        from pydantic_ai import Agent
        from pydantic_ai.models.vertexai import VertexAIModel

        project_id = os.getenv("VERTEX_PROJECT_ID") or os.getenv("BQ_PROJECT")
        region = os.getenv("VERTEX_LOCATION", "us-east1")

        model = VertexAIModel("gemini-1.5-pro", project_id=project_id, region=region)
        agent = Agent(model)
        result = agent.run_sync("Say hello in exactly one word.")
        assert result.data, "Expected non-empty response from Gemini"
        assert len(result.data) > 0


# ---------------------------------------------------------------------------
# 3. Embedding factory integration
# ---------------------------------------------------------------------------

class TestEmbeddingFactory:
    """Test that the embedding factory correctly routes to Vertex."""

    def test_resolve_backend_vertex(self):
        os.environ["EMBEDDING_BACKEND"] = "vertex"
        os.environ["VERTEX_EMBEDDING_MODEL"] = "text-embedding-005"
        from utils import resolve_embedding_backend_and_model
        backend, model = resolve_embedding_backend_and_model()
        assert backend == "vertex"
        assert "text-embedding-005" in model

    def test_create_embedding_function_vertex(self):
        _skip_if_no_adc()
        os.environ["EMBEDDING_BACKEND"] = "vertex"
        os.environ["VERTEX_EMBEDDING_MODEL"] = "text-embedding-005"
        from utils import create_embedding_function, _EMBEDDING_FUNCTION_CACHE
        # Clear cache to force fresh creation
        _EMBEDDING_FUNCTION_CACHE.clear()
        fn = create_embedding_function()
        assert fn is not None
        assert "vertex" in fn.name().lower() or "vertex" in type(fn).__name__.lower() or "lru" in type(fn).__name__.lower()


# ---------------------------------------------------------------------------
# 4. No-OpenAI-key required
# ---------------------------------------------------------------------------

class TestNoOpenAIRequired:
    """Verify that Vertex-configured startup does not require OPENAI_API_KEY."""

    def test_key_validation_passes_without_openai_key(self):
        # Remove OpenAI key from environment
        old_key = os.environ.pop("OPENAI_API_KEY", None)
        os.environ["EMBEDDING_BACKEND"] = "vertex"
        os.environ["CAL_MODEL_NAME"] = "gemini-1.5-pro"
        try:
            from utils import sanitize_and_validate_openai_key, get_key_diagnostics
            sanitize_and_validate_openai_key()
            diag = get_key_diagnostics()
            assert diag["valid"] is True, f"Expected valid=True when Vertex backends are active, got: {diag}"
        finally:
            # Restore original key
            if old_key is not None:
                os.environ["OPENAI_API_KEY"] = old_key

    def test_compute_fingerprint_vertex(self):
        os.environ["EMBEDDING_BACKEND"] = "vertex"
        os.environ["VERTEX_EMBEDDING_MODEL"] = "text-embedding-005"
        from utils import compute_embedding_fingerprint
        fp = compute_embedding_fingerprint()
        assert fp["backend"] == "vertex"
        assert fp["dim"] == 768
