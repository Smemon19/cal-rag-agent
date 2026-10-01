"""
Policy Badger - Raymond — FastAPI server with session-based auth.
Run locally: uvicorn policy_badger.web.app:app --reload
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
from fastapi import Depends, FastAPI, File, Form, Request, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.middleware.sessions import SessionMiddleware

# Load .env using explicit path so uvicorn always finds it regardless of CWD.
_APP_DIR = Path(__file__).resolve().parent
_REPO_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(dotenv_path=_REPO_ROOT / ".env", override=True)

from policy_badger.engine.service import answer_policy_question
from policy_badger.ingestion.admin_input_pipeline import (
    create_submission, extract_submission, validate_submission,
    generate_clarification_questions, preview_submission, publish_submission,
    _submissions
)

from policy_badger.engine.auth_db import (
    revalidate_session_user,
    authenticate_user, update_last_login, create_user,
    get_user_by_username, get_current_session_user, require_roles,
    require_login, list_users_for_admin,
    create_managed_user, reset_managed_user_password,
    deactivate_managed_user, reactivate_managed_user, verify_user_password,
    update_managed_user
)
from policy_badger.engine import audit_read, conversations_db, policy_admin_db, settings_db
from policy_badger.ingestion import document_upload
from policy_badger.engine.db import run_query, execute

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
app = FastAPI(title="Policy Badger - Raymond", docs_url=None, redoc_url=None)
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY, max_age=60 * 60 * 8)  # 8-hour session

PUBLIC_DIR = _APP_DIR / "static"
ASSETS_DIR = PUBLIC_DIR / "assets"

# Shared stylesheet/scripts only. HTML templates are NOT mounted — they stay
# behind the per-route session checks below.
app.mount("/assets", StaticFiles(directory=str(ASSETS_DIR)), name="assets")


@app.middleware("http")
async def revalidate_assets(request: Request, call_next):
    """Make browsers revalidate shared JS/CSS so a deploy never pairs new HTML
    with a stale cached badger.js. The ETag keeps unchanged files a 304."""
    response = await call_next(request)
    if request.url.path.startswith("/assets/"):
        response.headers["Cache-Control"] = "no-cache"
    return response

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

    company_name = os.environ.get("SEED_COMPANY_NAME", "Raymond Global")
    try:
        # Check or create default company
        company_rows = run_query("SELECT id FROM companies WHERE name = %s LIMIT 1", (company_name,))
        if company_rows:
            company_id = company_rows[0]["id"]
        else:
            from policy_badger.engine.db import transaction
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
def session_is_still_valid(request: Request) -> bool:
    """
    Re-check a page session against the database.

    The HTML routes read request.session directly rather than going through
    require_login, so without this a deactivated user keeps browsing until the
    cookie expires. Returns False once the account is gone or disabled, having
    already cleared the session.
    """
    user = get_current_session_user(request)
    if not user:
        return False
    try:
        revalidate_session_user(request, user)
        return True
    except HTTPException:
        return False


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
    if not session_is_still_valid(request):
        return RedirectResponse("/login", status_code=302)
    return HTMLResponse((PUBLIC_DIR / "index.html").read_text(encoding="utf-8"))

class AskRequest(BaseModel):
    question: str
    conversation_id: Optional[str] = None

@app.post("/ask")
async def ask(req: AskRequest, request: Request):
    if not request.session.get("user"):
        return JSONResponse({"error": "Not authenticated."}, status_code=401)
    if request.session.get("must_reset_password"):
        return JSONResponse({"error": "Password reset required."}, status_code=403)
    if not session_is_still_valid(request):
        return JSONResponse({"error": "Your session is no longer valid. Please sign in again."}, status_code=401)

    question = (req.question or "").strip()
    if not question:
        return JSONResponse({"error": "Question is empty."}, status_code=400)

    user_id = request.session.get("user_id")

    # Answer preferences are read server-side from the signed-in user rather
    # than trusted from the request body.
    try:
        user_settings = settings_db.get_user_settings(user_id) if user_id else {}
    except Exception:
        user_settings = {}

    try:
        result = answer_policy_question(question, settings=user_settings)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

    answer = result.get("answer", "")
    rows_found = result.get("row_count", 0)

    # History is best-effort: a persistence failure must not lose the answer
    # the user just waited for.
    conversation_id = req.conversation_id
    if user_id:
        try:
            conversation_id = conversations_db.append_exchange(
                user_id=user_id,
                conversation_id=req.conversation_id,
                question=question,
                answer=answer,
                rows_found=rows_found,
            )
        except Exception as e:
            logging.warning("Could not persist conversation: %s", e)

    return {
        "answer": answer,
        "rows_found": rows_found,
        "conversation_id": conversation_id,
    }

@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request):
    if not request.session.get("user"):
        return RedirectResponse("/login", status_code=302)
    if request.session.get("must_reset_password"):
        return RedirectResponse("/change-password", status_code=302)
    if not session_is_still_valid(request):
        return RedirectResponse("/login", status_code=302)
    return HTMLResponse((PUBLIC_DIR / "settings.html").read_text(encoding="utf-8"))

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

    if not session_is_still_valid(request):
        return RedirectResponse("/login", status_code=302)

    # Read the role after revalidation so a demotion is honoured immediately.
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

    try:
        preview_data = preview_submission(sub)
    except Exception as e:
        logging.exception("Policy preview failed")
        return JSONResponse({"error": f"Could not generate preview: {e}"}, status_code=502)
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
        policy_id = publish_submission(sub, published_by=user.get("username"))
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
_CHANGE_PW_PAGE = (PUBLIC_DIR / "change_password.html").read_text(encoding="utf-8")

def _change_pw_error(msg: str) -> HTMLResponse:
    html = _CHANGE_PW_PAGE.replace(
        "<!--ERROR-->",
        f'<div class="error-box">{msg}</div>'
    )
    return HTMLResponse(html, status_code=400)

@app.get("/change-password", response_class=HTMLResponse)
async def change_password_page(request: Request):
    user = get_current_session_user(request)
    if not user:
        return RedirectResponse("/login", status_code=302)
    return HTMLResponse(_CHANGE_PW_PAGE)

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
        return _change_pw_error("Passwords do not match. Please try again.")

    if len(new_password) < 8:
        return _change_pw_error("New password must be at least 8 characters.")

    user_id = user["user_id"]
    if user_id == "legacy":
        return _change_pw_error("Legacy users cannot change their password here.")

    try:
        rows = run_query("SELECT password_hash FROM users WHERE id = %s", (user_id,))
        if not rows:
            return RedirectResponse("/login", status_code=302)

        if not verify_user_password(current_password, rows[0]["password_hash"]):
            return _change_pw_error("Incorrect temporary password. Please try again.")

        new_hashed = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        execute("UPDATE users SET password_hash = %s, must_reset_password = false WHERE id = %s", (new_hashed, user_id))

        request.session["must_reset_password"] = False
        role = user.get("role")
        return RedirectResponse("/admin" if role in ("super_admin", "manager_admin") else "/", status_code=302)
    except Exception as e:
        return _change_pw_error("Something went wrong. Please try again.")

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

@app.post("/api/admin/users")
async def api_admin_create_user(req: CreateUserReq, request: Request):
    user = require_roles(request, {"manager_admin", "super_admin"})
    try:
        created = create_managed_user(
            current_user=user,
            username=req.username,
            password=req.password,
            role=req.role,
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

class UpdateUserReq(BaseModel):
    role: Optional[str] = None
    email: Optional[str] = None

class SettingsReq(BaseModel):
    settings: dict

class RenameConversationReq(BaseModel):
    title: str

class UpdatePolicyReq(BaseModel):
    updates: dict

class ArchivePolicyReq(BaseModel):
    reason: Optional[str] = None

class SupersedePolicyReq(BaseModel):
    replacement_policy_id: str

@app.patch("/api/admin/users/{user_id}")
async def api_admin_update_user(user_id: str, req: UpdateUserReq, request: Request):
    user = require_roles(request, {"manager_admin", "super_admin"})
    try:
        return update_managed_user(user, user_id, role=req.role, email=req.email)
    except HTTPException as he:
        raise he
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# ── Answer Settings ───────────────────────────────────────────────────────────
@app.get("/api/settings")
async def api_get_settings(request: Request):
    user = require_login(request)
    try:
        return {"settings": settings_db.get_user_settings(user["user_id"])}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.put("/api/settings")
async def api_save_settings(req: SettingsReq, request: Request):
    user = require_login(request)
    try:
        saved = settings_db.save_user_settings(user["user_id"], req.settings)
        return {"settings": saved}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# ── Conversation History ──────────────────────────────────────────────────────
@app.get("/api/conversations")
async def api_list_conversations(request: Request):
    user = require_login(request)
    try:
        return {"conversations": conversations_db.list_conversations(user["user_id"])}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/api/conversations/{conversation_id}")
async def api_get_conversation(conversation_id: str, request: Request):
    user = require_login(request)
    try:
        return conversations_db.get_conversation(user["user_id"], conversation_id)
    except HTTPException as he:
        raise he
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.patch("/api/conversations/{conversation_id}")
async def api_rename_conversation(conversation_id: str, req: RenameConversationReq, request: Request):
    user = require_login(request)
    try:
        conversations_db.rename_conversation(user["user_id"], conversation_id, req.title)
        return {"message": "Conversation renamed."}
    except HTTPException as he:
        raise he
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.delete("/api/conversations/{conversation_id}")
async def api_delete_conversation(conversation_id: str, request: Request):
    user = require_login(request)
    try:
        conversations_db.delete_conversation(user["user_id"], conversation_id)
        return {"message": "Conversation deleted."}
    except HTTPException as he:
        raise he
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# ── Browse ────────────────────────────────────────────────────────────────────
@app.get("/api/topics")
async def api_topics(request: Request):
    """Policy topics for the sidebar folders. Available to any signed-in user."""
    require_login(request)
    try:
        return {"topics": policy_admin_db.list_topics()}
    except Exception:
        # The rail degrades to no folders rather than failing the page.
        return {"topics": []}

# ── Policy Library ────────────────────────────────────────────────────────────
@app.get("/api/admin/policies")
async def api_list_policies(
    request: Request,
    search: Optional[str] = None,
    status: Optional[str] = None,
    topic: Optional[str] = None,
    page: int = 1,
    page_size: int = 25,
):
    require_roles(request, {"manager_admin", "super_admin"})
    try:
        return policy_admin_db.list_policies(
            search=search, status=status, topic=topic, page=page, page_size=page_size
        )
    except HTTPException as he:
        raise he
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/api/admin/policies/topics")
async def api_list_policy_topics(request: Request):
    require_roles(request, {"manager_admin", "super_admin"})
    try:
        return {"topics": policy_admin_db.list_topics(), "editable_fields": policy_admin_db.editable_fields()}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/api/admin/policies/{policy_id}")
async def api_get_policy(policy_id: str, request: Request):
    require_roles(request, {"manager_admin", "super_admin"})
    try:
        return {"policy": policy_admin_db.get_policy(policy_id)}
    except HTTPException as he:
        raise he
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.patch("/api/admin/policies/{policy_id}")
async def api_update_policy(policy_id: str, req: UpdatePolicyReq, request: Request):
    user = require_roles(request, {"manager_admin", "super_admin"})
    try:
        return {"policy": policy_admin_db.update_policy(user["user_id"], policy_id, req.updates)}
    except HTTPException as he:
        raise he
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/api/admin/policies/{policy_id}/archive")
async def api_archive_policy(policy_id: str, req: ArchivePolicyReq, request: Request):
    # Retiring a policy changes what every employee is told, so it is
    # super-admin only.
    user = require_roles(request, {"super_admin"})
    try:
        policy_admin_db.archive_policy(user["user_id"], policy_id, req.reason or "")
        return {"message": "Policy archived."}
    except HTTPException as he:
        raise he
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/api/admin/policies/{policy_id}/restore")
async def api_restore_policy(policy_id: str, request: Request):
    user = require_roles(request, {"super_admin"})
    try:
        policy_admin_db.restore_policy(user["user_id"], policy_id)
        return {"message": "Policy restored."}
    except HTTPException as he:
        raise he
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/api/admin/policies/{policy_id}/supersede")
async def api_supersede_policy(policy_id: str, req: SupersedePolicyReq, request: Request):
    user = require_roles(request, {"super_admin"})
    try:
        policy_admin_db.supersede_policy(user["user_id"], policy_id, req.replacement_policy_id)
        return {"message": "Policy superseded."}
    except HTTPException as he:
        raise he
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# ── Document Upload & Review ──────────────────────────────────────────────────
class PublishSelectedReq(BaseModel):
    extraction_ids: list[str]

@app.post("/api/admin/ingest/upload")
async def api_ingest_upload(
    request: Request,
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
):
    """Extract policies from an uploaded document and return them for review."""
    user = require_roles(request, {"manager_admin", "super_admin"})
    content = await file.read()
    try:
        return document_upload.ingest_upload(
            filename=file.filename or "upload",
            content=content,
            title=title,
            triggered_by=user["username"],
        )
    except HTTPException as he:
        raise he
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/api/admin/ingest/{batch_id}/review")
async def api_ingest_review(batch_id: str, request: Request):
    require_roles(request, {"manager_admin", "super_admin"})
    try:
        return document_upload.get_review(batch_id)
    except HTTPException as he:
        raise he
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/api/admin/ingest/{batch_id}/publish")
async def api_ingest_publish(batch_id: str, req: PublishSelectedReq, request: Request):
    user = require_roles(request, {"manager_admin", "super_admin"})
    try:
        return document_upload.publish_selected(
            batch_id=batch_id,
            extraction_ids=req.extraction_ids,
            published_by=user["user_id"],
        )
    except HTTPException as he:
        raise he
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/api/admin/ingest/{batch_id}/discard")
async def api_ingest_discard(batch_id: str, request: Request):
    require_roles(request, {"manager_admin", "super_admin"})
    try:
        return document_upload.discard_batch(batch_id)
    except HTTPException as he:
        raise he
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# ── Audit Trail ───────────────────────────────────────────────────────────────
@app.get("/api/admin/audit")
async def api_list_audit(
    request: Request,
    source: Optional[str] = None,
    action: Optional[str] = None,
    actor: Optional[str] = None,
    page: int = 1,
    page_size: int = 50,
):
    # The audit trail names who did what to whom; super admins only.
    require_roles(request, {"super_admin"})
    if source and source not in audit_read.VALID_SOURCES:
        return JSONResponse({"error": "Unknown audit source."}, status_code=400)
    try:
        return audit_read.list_audit_events(
            source=source, action=action, actor=actor, page=page, page_size=page_size
        )
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/api/admin/audit/actions")
async def api_list_audit_actions(request: Request):
    require_roles(request, {"super_admin"})
    try:
        return {"actions": audit_read.list_audit_actions()}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
