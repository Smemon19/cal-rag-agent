"""
Simple FastAPI server for the Policy Badger Q&A interface.
Run with: uvicorn app:app --reload
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Load .env using explicit path so uvicorn finds it regardless of CWD
_ENV_FILE = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=_ENV_FILE, override=True)

# Ensure policy_engine is importable when running from repo root
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from policy_engine.service import answer_policy_question

app = FastAPI(title="Policy Badger")

# Serve static files (the single-page UI)
PUBLIC_DIR = Path(__file__).resolve().parent / "public"
PUBLIC_DIR.mkdir(exist_ok=True)


class AskRequest(BaseModel):
    question: str


@app.post("/ask")
async def ask(req: AskRequest):
    question = (req.question or "").strip()
    if not question:
        return JSONResponse({"error": "Question is empty."}, status_code=400)

    try:
        result = answer_policy_question(question)
        return {
            "answer": result.get("answer", ""),
            "rows_found": result.get("row_count", 0),
        }
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/", response_class=HTMLResponse)
async def index():
    html_file = PUBLIC_DIR / "index.html"
    return HTMLResponse(html_file.read_text(encoding="utf-8"))
