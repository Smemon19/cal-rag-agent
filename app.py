"""
Policy Badger — FastAPI server with session-based auth.
Run locally: uvicorn app:app --reload
"""

from __future__ import annotations

import json
import os
import sys
import logging
from pathlib import Path
from typing import Optional

import bcrypt
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Form, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel
from starlette.middleware.sessions import SessionMiddleware

# Load .env using explicit path so uvicorn always finds it regardless of CWD
_ROOT = Path(__file__).resolve().parent
load_dotenv(dotenv_path=_ROOT / ".env", override=True)
sys.path.insert(0, str(_ROOT))

from policy_engine.service import answer_policy_question
from adaptive_ingestion.admin_input_pipeline import (
    create_submission, extract_submission, validate_submission,
    generate_clarification_questions, preview_submission, publish_submission,
    _submissions
)

from policy_engine.auth_db import (
    authenticate_user, update_last_login, create_user,
    get_user_by_username, get_current_session_user, require_roles,
    require_login, list_users_for_admin, list_companies_for_admin,
    create_company, create_managed_user, reset_managed_user_password,
    deactivate_managed_user, reactivate_managed_user, verify_user_password
)
from policy_engine.db import run_query, execute

# ── Production safety guard ──────────────────────────────────────────────────
ENABLE_LEGACY_APP_USERS_AUTH = os.environ.get("ENABLE_LEGACY_APP_USERS_AUTH", "false").lower() == "true"
ALLOW_PROD_LEGACY_AUTH = os.environ.get("ALLOW_PROD_LEGACY_AUTH", "false").lower() == "true"
APP_ENV = os.environ.get("APP_ENV", os.environ.get("ENVIRONMENT", "development")).lower()

if APP_ENV == "production" and ENABLE_LEGACY_APP_USERS_AUTH and not ALLOW_PROD_LEGACY_AUTH:
    logging.critical("CRITICAL: ENABLE_LEGACY_APP_USERS_AUTH is true in production, but ALLOW_PROD_LEGACY_AUTH is not set! Halting startup for safety.")
    sys.exit(1)

# ── Legacy Auth Config ────────────────────────────────────────────────────────
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

# ── Seeding Logic ─────────────────────────────────────────────────────────────
@app.on_event("startup")
def _seed_super_admin():
    username = os.environ.get("SEED_SUPER_ADMIN_USERNAME")
    password = os.environ.get("SEED_SUPER_ADMIN_PASSWORD")
    if not username or not password:
        return

    # check if user already exists
    try:
        user = get_user_by_username(username)
        if user:
            logging.info(f"Seed super admin '{username}' already exists.")
            return
    except Exception as e:
        logging.error(f"Error checking existing seed super admin: {str(e)}")
        return

    company_name = os.environ.get("SEED_COMPANY_NAME", "Default Company")
    try:
        # Check or create default company
        company_rows = run_query("SELECT id FROM companies WHERE name = %s LIMIT 1", (company_name,))
        if company_rows:
            company_id = company_rows[0]["id"]
        else:
            from policy_engine.db import transaction
            with transaction() as cur:
                cur.execute("INSERT INTO companies (name) VALUES (%s) RETURNING id", (company_name,))
                rows = cur.fetchall()
            company_id = dict(rows[0])["id"] if rows else None

        email = os.environ.get("SEED_SUPER_ADMIN_EMAIL")
        user_id = create_user(
            username=username,
            password=password,
            role="super_admin",
            company_id=company_id,
            email=email,
            must_reset_password=True
        )
        logging.info(f"Seed super admin '{username}' created successfully.")
    except Exception as e:
        logging.error(f"Failed to seed super admin: {str(e)}")

# ── Auth dependency ───────────────────────────────────────────────────────────
def require_auth(request: Request):
    if not request.session.get("user"):
        raise _redirect_to_login()
    if request.session.get("must_reset_password"):
        raise RedirectResponse("/change-password", status_code=302)
    return request.session["user"]

def _redirect_to_login():
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
    # Try database-backed auth first
    try:
        user = authenticate_user(username, password)
        if user:
            if not user.get("is_active", True):
                return HTMLResponse(
                    _LOGIN_PAGE.replace("<!--ERROR-->",
                        '<p class="error">Account is deactivated.</p>'),
                    status_code=401,
                )

            update_last_login(user["id"])
            request.session["user_id"] = str(user["id"])
            request.session["username"] = user["username"]
            request.session["role"] = user["role"]
            request.session["company_id"] = str(user["company_id"]) if user.get("company_id") else None
            request.session["must_reset_password"] = bool(user.get("must_reset_password"))

            # backward compatibility
            request.session["user"] = user["username"]

            if user.get("must_reset_password"):
                return RedirectResponse("/change-password", status_code=302)

            return RedirectResponse("/", status_code=302)
    except Exception as e:
        logging.error(f"DB Auth Error: {str(e)}")

    # Legacy fallback
    if ENABLE_LEGACY_APP_USERS_AUTH:
        if _check_password(username, password):
            logging.warning("ENABLE_LEGACY_APP_USERS_AUTH is true. Legacy APP_USERS will receive 'super_admin' access. USE ONLY FOR EMERGENCY RECOVERY.")
            request.session["user_id"] = "legacy"
            request.session["username"] = username
            request.session["role"] = "super_admin"
            request.session["company_id"] = None
            request.session["must_reset_password"] = False
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
    if request.session.get("must_reset_password"):
        return RedirectResponse("/change-password", status_code=302)
    return HTMLResponse((PUBLIC_DIR / "index.html").read_text(encoding="utf-8"))

class AskRequest(BaseModel):
    question: str

@app.post("/ask")
async def ask(req: AskRequest, request: Request):
    if not request.session.get("user"):
        return JSONResponse({"error": "Not authenticated."}, status_code=401)
    if request.session.get("must_reset_password"):
        return JSONResponse({"error": "Password reset required."}, status_code=403)

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

# ── Admin Routes ──────────────────────────────────────────────────────────────
@app.get("/admin", response_class=HTMLResponse)
async def admin_page(request: Request):
    if not request.session.get("user"):
        return RedirectResponse("/login", status_code=302)
    if request.session.get("must_reset_password"):
        return RedirectResponse("/change-password", status_code=302)

    role = request.session.get("role")
    if role == "employee":
        return HTMLResponse(
            "<h1>403 Forbidden</h1><p>You do not have permission to access the admin area.</p><a href='/'>Go Back</a>",
            status_code=403
        )
    return HTMLResponse((PUBLIC_DIR / "admin.html").read_text(encoding="utf-8"))

class ExtractRequest(BaseModel):
    title: str
    raw_text: str

@app.post("/api/admin/extract")
async def api_admin_extract(req: ExtractRequest, request: Request):
    user = require_roles(request, {"manager_admin", "super_admin"})
    if not user:
        return JSONResponse({"error": "Not authenticated."}, status_code=401)
    if request.session.get("must_reset_password"):
        return JSONResponse({"error": "Password reset required."}, status_code=403)

    username = user["username"]
    sub = create_submission(req.title, req.raw_text, submitted_by=username)
    extract_submission(sub, use_llm=True)
    validate_submission(sub)

    questions = []
    if sub.status == "needs_clarification":
        questions = generate_clarification_questions(sub)

    return {
        "id": sub.id,
        "status": sub.status,
        "confidence": sub.confidence,
        "extracted_json": sub.extracted_json,
        "questions": questions
    }

class ClarifyRequest(BaseModel):
    id: str
    clarification: str

@app.post("/api/admin/clarify")
async def api_admin_clarify(req: ClarifyRequest, request: Request):
    user = require_roles(request, {"manager_admin", "super_admin"})
    if not user:
        return JSONResponse({"error": "Not authenticated."}, status_code=401)
    if request.session.get("must_reset_password"):
        return JSONResponse({"error": "Password reset required."}, status_code=403)

    sub = _submissions.get(req.id)
    if not sub:
        return JSONResponse({"error": "Submission not found."}, status_code=404)

    sub.raw_text += f"\nClarification: {req.clarification}"
    extract_submission(sub, use_llm=True)
    validate_submission(sub)

    questions = []
    if sub.status == "needs_clarification":
        questions = generate_clarification_questions(sub)

    return {
        "id": sub.id,
        "status": sub.status,
        "confidence": sub.confidence,
        "extracted_json": sub.extracted_json,
        "questions": questions
    }

class EditRequest(BaseModel):
    id: str
    edited_json: dict

def _validate_admin_policy_item(item: dict) -> list[str]:
    missing = []
    if not item.get("topic"):
        missing.append("topic")
    if not item.get("source_quote"):
        missing.append("source_quote")
    if not item.get("action_text") and not item.get("condition_text"):
        missing.append("action_text OR condition_text")
    return missing

@app.post("/api/admin/preview")
async def api_admin_preview(req: EditRequest, request: Request):
    user = require_roles(request, {"manager_admin", "super_admin"})
    if not user:
        return JSONResponse({"error": "Not authenticated."}, status_code=401)
    if request.session.get("must_reset_password"):
        return JSONResponse({"error": "Password reset required."}, status_code=403)

    sub = _submissions.get(req.id)
    if not sub:
        return JSONResponse({"error": "Submission not found."}, status_code=404)

    sub.extracted_json = req.edited_json

    item = req.edited_json.get("item", {}) if isinstance(req.edited_json, dict) else {}
    if not item and "topic" in req.edited_json:
        item = req.edited_json

    missing_fields = _validate_admin_policy_item(item)
    if missing_fields:
        return JSONResponse({"error": f"Missing required fields: {', '.join(missing_fields)}"}, status_code=400)

    preview_data = preview_submission(sub)
    if not preview_data:
        return JSONResponse({"error": "Could not generate preview."}, status_code=400)

    return preview_data

@app.post("/api/admin/publish")
async def api_admin_publish(req: EditRequest, request: Request):
    user = require_roles(request, {"manager_admin", "super_admin"})
    if not user:
        return JSONResponse({"error": "Not authenticated."}, status_code=401)
    if request.session.get("must_reset_password"):
        return JSONResponse({"error": "Password reset required."}, status_code=403)

    sub = _submissions.get(req.id)
    if not sub:
        return JSONResponse({"error": "Submission not found."}, status_code=404)

    sub.extracted_json = req.edited_json

    item = req.edited_json.get("item", {}) if isinstance(req.edited_json, dict) else {}
    if not item and "topic" in req.edited_json:
        item = req.edited_json

    missing_fields = _validate_admin_policy_item(item)
    if missing_fields:
        return JSONResponse({"error": f"Cannot publish. Missing required fields: {', '.join(missing_fields)}"}, status_code=400)

    try:
        policy_id = publish_submission(sub)
        return {"policy_id": policy_id, "message": "Success"}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# ── Auth Info Endpoint ────────────────────────────────────────────────────────
@app.get("/api/auth/me")
async def api_auth_me(request: Request):
    user = get_current_session_user(request)
    if not user:
        return JSONResponse({"error": "Not authenticated"}, status_code=401)
    return user

# ── Password Change Routes ────────────────────────────────────────────────────
@app.get("/change-password", response_class=HTMLResponse)
async def change_password_page(request: Request):
    user = get_current_session_user(request)
    if not user:
        return RedirectResponse("/login", status_code=302)

    change_html = (PUBLIC_DIR / "change_password.html")
    if change_html.exists():
        return HTMLResponse(change_html.read_text(encoding="utf-8"))

    # Simple fallback HTML if file is missing
    fallback = """
        <html><body><h2>Change Password Required</h2>
        <form method='POST' action='/change-password'>
            <input type='password' name='current_password' placeholder='Current Password' required><br>
            <input type='password' name='new_password' placeholder='New Password' required><br>
            <input type='password' name='confirm_password' placeholder='Confirm New Password' required><br>
            <button type='submit'>Change Password</button>
        </form></body></html>
        """
    return HTMLResponse(fallback)

@app.post("/change-password")
async def change_password_post(
    request: Request,
    current_password: str = Form(...),
    new_password: str = Form(...),
    confirm_password: str = Form(...)
):
    user = get_current_session_user(request)
    if not user:
        return RedirectResponse("/login", status_code=302)

    if new_password != confirm_password:
        return HTMLResponse("Passwords do not match. <a href='/change-password'>Try again</a>", status_code=400)

    user_id = user["user_id"]
    if user_id == "legacy":
        return HTMLResponse("Legacy users cannot change password via this interface.", status_code=400)

    # verify user and update
    try:
        rows = run_query("SELECT password_hash FROM users WHERE id = %s", (user_id,))
        if not rows:
            return RedirectResponse("/login", status_code=302)

        hashed = rows[0]["password_hash"]
        if not verify_user_password(current_password, hashed):
            return HTMLResponse("Incorrect current password. <a href='/change-password'>Try again</a>", status_code=400)

        new_hashed = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        execute("UPDATE users SET password_hash = %s, must_reset_password = false WHERE id = %s", (new_hashed, user_id))

        # update session
        request.session["must_reset_password"] = False
        return RedirectResponse("/", status_code=302)
    except Exception as e:
        return HTMLResponse(f"Error: {str(e)}", status_code=500)

# ── User & Company Management ─────────────────────────────────────────────────
@app.get("/api/admin/users")
async def api_admin_get_users(request: Request):
    user = require_roles(request, {"manager_admin", "super_admin"})
    try:
        users = list_users_for_admin(user)
        return {"users": users}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

class CreateUserReq(BaseModel):
    username: str
    password: str
    role: str
    email: Optional[str] = None
    company_id: Optional[str] = None

@app.post("/api/admin/users")
async def api_admin_create_user(req: CreateUserReq, request: Request):
    user = require_roles(request, {"manager_admin", "super_admin"})
    try:
        created = create_managed_user(
            current_user=user,
            username=req.username,
            password=req.password,
            role=req.role,
            company_id=req.company_id,
            email=req.email,
            must_reset_password=True
        )
        return created
    except HTTPException as he:
        raise he
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

class ResetPassReq(BaseModel):
    new_password: str

@app.post("/api/admin/users/{user_id}/reset-password")
async def api_admin_reset_user_pass(user_id: str, req: ResetPassReq, request: Request):
    user = require_roles(request, {"manager_admin", "super_admin"})
    try:
        reset_managed_user_password(user, user_id, req.new_password)
        return {"message": "Password reset successfully"}
    except HTTPException as he:
        raise he
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/api/admin/users/{user_id}/deactivate")
async def api_admin_deactivate_user(user_id: str, request: Request):
    user = require_roles(request, {"manager_admin", "super_admin"})
    try:
        deactivate_managed_user(user, user_id)
        return {"message": "User deactivated successfully"}
    except HTTPException as he:
        raise he
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/api/admin/users/{user_id}/reactivate")
async def api_admin_reactivate_user(user_id: str, request: Request):
    user = require_roles(request, {"manager_admin", "super_admin"})
    try:
        reactivate_managed_user(user, user_id)
        return {"message": "User reactivated successfully"}
    except HTTPException as he:
        raise he
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/api/admin/companies")
async def api_admin_get_companies(request: Request):
    user = require_roles(request, {"super_admin"})
    try:
        companies = list_companies_for_admin(user)
        return {"companies": companies}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

class CreateCompanyReq(BaseModel):
    name: str
    slug: Optional[str] = None

@app.post("/api/admin/companies")
async def api_admin_create_company(req: CreateCompanyReq, request: Request):
    user = require_roles(request, {"super_admin"})
    try:
        created = create_company(user, req.name, req.slug)
        return created
    except HTTPException as he:
        raise he
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
