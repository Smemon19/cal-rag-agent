"""
Chat model access for the question-answering path (planner + formatter).

Calls Gemini through Vertex AI using the runtime's Google credentials, so the
app bills through GCP and needs no third-party API key. On Cloud Run that is
the service account; locally it is `gcloud auth application-default login`.
"""

from __future__ import annotations

import os
import threading

DEFAULT_MODEL = "gemini-2.5-flash"

_client = None
_client_lock = threading.Lock()


def _get_client():
    global _client
    if _client is not None:
        return _client
    with _client_lock:
        if _client is None:
            import google.auth
            from google import genai

            _, detected_project = google.auth.default()
            project = (
                os.getenv("VERTEX_PROJECT_ID")
                or os.getenv("GOOGLE_CLOUD_PROJECT")
                or detected_project
            )
            if not project:
                raise RuntimeError(
                    "No Google Cloud project for Gemini. Set VERTEX_PROJECT_ID or run "
                    "`gcloud auth application-default login`."
                )
            location = os.getenv("VERTEX_LOCATION", "us-central1")
            _client = genai.Client(vertexai=True, project=project, location=location)
    return _client


def generate(system: str, user: str, *, temperature: float, json_output: bool = False) -> str:
    """
    Run one system + user prompt and return the model's text.

    json_output asks Gemini to return a JSON body, which the planner parses.
    """
    from google.genai import types

    config = types.GenerateContentConfig(
        system_instruction=system,
        temperature=temperature,
        response_mime_type="application/json" if json_output else None,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    response = _get_client().models.generate_content(
        model=os.getenv("CHAT_LLM_MODEL") or DEFAULT_MODEL,
        contents=user,
        config=config,
    )
    text = (response.text or "").strip()
    if not text:
        raise ValueError("The language model returned no content.")
    return text
