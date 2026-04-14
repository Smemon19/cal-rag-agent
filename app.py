"""
Policy Badger — FastAPI server with session-based auth.
Run locally: uvicorn app:app --reload
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import bcrypt
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel
from starlette.middleware.sessions import SessionMiddleware

# Load .env using explicit path so uvicorn always finds it regardless of CWD
_ROOT = Path(__file__).resolve().parent
load_dotenv(dotenv_path=_ROOT / ".env", override=True)
sys.path.insert(0, str(_ROOT))

from policy_engine.service import answer_policy_question

# ── Auth config ───────────────────────────────────────────────────────────────
# APP_USERS is a JSON string: {"username": "plaintext_password", ...}
# Falls back to a single user if APP_USERS is not set.
SECRET_KEY = os.environ.get("APP_SECRET_KEY", "change-me-in-production-32chars!!")

def _load_users() -> dict[str, bytes]:
    """Return {username: bcrypt_hash} for all configured users."""
    raw = os.environ.get("APP_USERS", "")
    if raw:
        try:
            pairs = json.loads(raw)
        except Exception:
            pairs = {}
    else:
        # Legacy single-user fallback
        pairs = {
            os.environ.get("APP_USER", "Test"): os.environ.get("APP_PASSWORD", "Test123!")
        }
    return {u: bcrypt.hashpw(p.encode(), bcrypt.gensalt()) for u, p in pairs.items()}

_USERS: dict[str, bytes] = _load_users()

def _check_password(username: str, plain: str) -> bool:
    h = _USERS.get(username)
    if not h:
        return False
    return bcrypt.checkpw(plain.encode(), h)

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(title="Policy Badger", docs_url=None, redoc_url=None)
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY, max_age=60 * 60 * 8)  # 8-hour session

PUBLIC_DIR = _ROOT / "public"
PUBLIC_DIR.mkdir(exist_ok=True)

# ── Auth dependency ───────────────────────────────────────────────────────────
def require_auth(request: Request):
    if not request.session.get("user"):
        raise _redirect_to_login()
    return request.session["user"]

def _redirect_to_login():
    from fastapi import HTTPException
    # We raise a redirect; FastAPI exception handlers will catch it
    return HTTPException(status_code=302, headers={"Location": "/login"})

# ── Login routes ──────────────────────────────────────────────────────────────
_LOGIN_PAGE = (PUBLIC_DIR / "login.html").read_text(encoding="utf-8")

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    if request.session.get("user"):
        return RedirectResponse("/", status_code=302)
    return HTMLResponse(_LOGIN_PAGE)

@app.post("/login")
async def login(request: Request, username: str = Form(...), password: str = Form(...)):
    if _check_password(username, password):
        request.session["user"] = username
        return RedirectResponse("/", status_code=302)
    return HTMLResponse(
        _LOGIN_PAGE.replace("<!--ERROR-->",
            '<p class="error">Incorrect username or password.</p>'),
        status_code=401,
    )

@app.get("/logout")
async def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/login", status_code=302)

# ── Main routes ───────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    if not request.session.get("user"):
        return RedirectResponse("/login", status_code=302)
    return HTMLResponse((PUBLIC_DIR / "index.html").read_text(encoding="utf-8"))

class AskRequest(BaseModel):
    question: str

@app.post("/ask")
async def ask(req: AskRequest, request: Request):
    if not request.session.get("user"):
        return JSONResponse({"error": "Not authenticated."}, status_code=401)

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

@app.get("/health")
async def health():
    return {"status": "ok"}
