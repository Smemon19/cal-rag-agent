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

from policy_engine.service import answer_policy_question

_WEB_DIR = Path(__file__).resolve().parent / "web"


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


app = Starlette(
    debug=False,
    routes=[
        Route("/", homepage, methods=["GET"]),
        Route("/ask", ask, methods=["POST"]),
    ],
)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8765, log_level="info")
