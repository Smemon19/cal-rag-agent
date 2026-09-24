import time

import bcrypt
from typing import Optional, Dict, Set
from fastapi import Request, HTTPException
from policy_badger.engine.db import run_query, execute

def get_user_by_username(username: str) -> Optional[Dict]:
    """
    Looks up an active user by username.
    Returns id, username, email, password_hash, role, company_id, must_reset_password, is_active.
    """
    sql = """
        SELECT id, username, email, password_hash, role, company_id, must_reset_password, is_active
        FROM users
        WHERE username = %s AND is_active = true
    """
    rows = run_query(sql, (username,))
    return rows[0] if rows else None

def verify_user_password(plain_password: str, password_hash: str) -> bool:
    """Uses bcrypt.checkpw to verify a plaintext password against a stored hash."""
    return bcrypt.checkpw(plain_password.encode('utf-8'), password_hash.encode('utf-8'))

def authenticate_user(username: str, password: str) -> Optional[Dict]:
    """
    Looks up an active user and verifies their password.
    Returns the user dict if valid and active, else None.
    """
    user = get_user_by_username(username)
    if not user:
        return None
    if verify_user_password(password, user['password_hash']):
        return user
    return None

def update_last_login(user_id: str) -> None:
    """Updates last_login_at = now() for the given user."""
    sql = "UPDATE users SET last_login_at = now() WHERE id = %s"
    execute(sql, (user_id,))

def create_user(
    username: str,
    password: str,
    role: str,
    company_id: Optional[str] = None,
    email: Optional[str] = None,
    created_by_user_id: Optional[str] = None,
    must_reset_password: bool = False
) -> str:
    """
    Creates a new user, hashing their password with bcrypt.
    Returns the newly created user's ID.
    """
    # Hash password securely
    password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    sql = """
        INSERT INTO users (
            username, email, password_hash, role, 
            company_id, created_by_user_id, must_reset_password
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s
        ) RETURNING id
    """
    # use transaction to get the RETURNING id and commit
    from policy_badger.engine.db import transaction
    with transaction() as cur:
        cur.execute(sql, (
            username, email, password_hash, role,
            company_id, created_by_user_id, must_reset_password
        ))
        rows = cur.fetchall()
    return str(dict(rows[0])['id']) if rows else ""

def get_current_session_user(request: Request) -> Optional[Dict]:
    """
    Reads from request.session.
    Returns dict containing user_id, username, role, company_id if present.
    """
    session = request.session
    if "user_id" in session and "username" in session and "role" in session:
        return {
            "user_id": session["user_id"],
            "username": session["username"],
            "role": session["role"],
            "company_id": session.get("company_id")
        }
    return None

def require_login(request: Request) -> Dict:
    """
    For API usage: Raises HTTPException 401 if no logged-in user.
    """
    user = get_current_session_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated.")
    return revalidate_session_user(request, user)


# How long a session may go without being re-checked against the database.
# Deactivating or demoting a user takes effect within this window rather than
# waiting out the full 8-hour cookie lifetime.
SESSION_REVALIDATE_SECONDS = 60


def revalidate_session_user(request: Request, user: Dict) -> Dict:
    """
    Re-read role and is_active from the database for a live session.

    The session cookie carries the role captured at login, so a user who was
    deactivated or demoted mid-session kept their old access until the cookie
    expired. This re-checks periodically and clears the session when the
    account is gone or disabled.

    The result is cached in the session for SESSION_REVALIDATE_SECONDS so a
    burst of requests does not mean a query each.
    """
    now = time.time()
    checked_at = request.session.get("revalidated_at", 0)
    if now - checked_at < SESSION_REVALIDATE_SECONDS:
        return user

    try:
        rows = run_query(
            "SELECT role, is_active, company_id FROM users WHERE id = %s",
            (user["user_id"],),
        )
    except Exception:
        # A database blip should not sign everybody out; keep the cached claims
        # and re-check on the next request.
        return user

    if not rows:
        request.session.clear()
        raise HTTPException(status_code=401, detail="Account no longer exists.")

    row = rows[0]
    if not row["is_active"]:
        request.session.clear()
        raise HTTPException(status_code=401, detail="Account has been deactivated.")

    # Pick up role or company changes without forcing a re-login.
    request.session["role"] = row["role"]
    request.session["company_id"] = str(row["company_id"]) if row["company_id"] else None
    request.session["revalidated_at"] = now

    user = dict(user)
    user["role"] = row["role"]
    user["company_id"] = request.session["company_id"]
    return user

def require_roles(request: Request, allowed_roles: Set[str]) -> Dict:
    """
    Raises HTTPException 401 if not logged in or no longer active.
    Raises HTTPException 403 if logged in but role not allowed.
    Returns the session user dict, with the role refreshed from the database.
    """
    # require_login already refreshed role/is_active from the database.
    user = require_login(request)
    if user["role"] not in allowed_roles:
        raise HTTPException(status_code=403, detail="Forbidden. Insufficient role.")
    return user

# ── Phase 2: User Management & Audit ─────────────────────────────────────────

import json

def log_audit_action(actor_user_id: Optional[str], target_user_id: Optional[str], action: str, details: dict):
    sql = """
        INSERT INTO user_audit_log (actor_user_id, target_user_id, action, details)
        VALUES (%s, %s, %s, %s)
    """
    execute(sql, (actor_user_id, target_user_id, action, json.dumps(details)))

def _get_raymond_global_company_id() -> Optional[str]:
    rows = run_query("SELECT id FROM companies WHERE name = 'Raymond Global' LIMIT 1", ())
    return str(rows[0]["id"]) if rows else None

def list_users_for_admin(current_user: dict) -> list[dict]:
    role = current_user.get("role")
    if role == "super_admin":
        sql = """
            SELECT u.id, u.username, u.email, u.role, u.is_active,
                   creator.username AS created_by_username
            FROM users u
            LEFT JOIN users creator ON u.created_by_user_id = creator.id
            ORDER BY u.role, u.username
        """
        return run_query(sql)
    elif role == "manager_admin":
        user_id = current_user.get("user_id")
        sql = """
            SELECT u.id, u.username, u.email, u.role, u.is_active
            FROM users u
            WHERE u.created_by_user_id = %s AND u.role = 'employee'
            ORDER BY u.username
        """
        return run_query(sql, (user_id,))
    else:
        raise HTTPException(status_code=403, detail="Forbidden")

def create_managed_user(
    current_user: dict,
    username: str,
    password: str,
    role: str,
    email: Optional[str] = None,
    must_reset_password: bool = True
) -> dict:
    curr_role = current_user.get("role")

    if curr_role == "employee":
        raise HTTPException(status_code=403, detail="Forbidden")

    if curr_role == "manager_admin":
        if role != "employee":
            raise HTTPException(status_code=403, detail="Manager can only create employee users.")

    if curr_role == "super_admin":
        if role not in ["employee", "manager_admin"]:
            raise HTTPException(status_code=403, detail="Cannot create super_admin through this endpoint.")

    company_id = _get_raymond_global_company_id()

    try:
        user_id = create_user(
            username=username,
            password=password,
            role=role,
            company_id=company_id,
            email=email,
            created_by_user_id=current_user["user_id"],
            must_reset_password=must_reset_password
        )
        log_audit_action(current_user["user_id"], user_id, "create_user", {"username": username, "role": role})
        return {
            "id": user_id, "username": username, "role": role,
            "email": email, "is_active": True
        }
    except Exception as e:
        if "unique constraint" in str(e).lower():
            raise HTTPException(status_code=400, detail="Username or email already exists.")
        raise HTTPException(status_code=500, detail="Database error creating user.")

def _verify_managed_user_target(current_user: dict, target_user_id: str) -> dict:
    rows = run_query("SELECT id, role, company_id FROM users WHERE id = %s", (target_user_id,))
    if not rows:
        raise HTTPException(status_code=404, detail="User not found.")
    target = rows[0]
    
    curr_role = current_user.get("role")
    
    if target["role"] == "super_admin":
        raise HTTPException(status_code=403, detail="Cannot modify super_admin through this endpoint.")
        
    if current_user["user_id"] == str(target["id"]):
        raise HTTPException(status_code=403, detail="Cannot perform this action on yourself.")
        
    if curr_role == "manager_admin":
        if target["role"] != "employee" or str(target["company_id"]) != str(current_user.get("company_id")):
            raise HTTPException(status_code=403, detail="Forbidden. Not authorized to manage this user.")
            
    return target

def reset_managed_user_password(current_user: dict, target_user_id: str, new_password: str) -> None:
    _verify_managed_user_target(current_user, target_user_id)
    password_hash = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    execute("UPDATE users SET password_hash = %s, must_reset_password = true WHERE id = %s", (password_hash, target_user_id))
    log_audit_action(current_user["user_id"], target_user_id, "reset_password", {})

def deactivate_managed_user(current_user: dict, target_user_id: str) -> None:
    _verify_managed_user_target(current_user, target_user_id)
    execute("UPDATE users SET is_active = false WHERE id = %s", (target_user_id,))
    log_audit_action(current_user["user_id"], target_user_id, "deactivate_user", {})

def reactivate_managed_user(current_user: dict, target_user_id: str) -> None:
    _verify_managed_user_target(current_user, target_user_id)
    execute("UPDATE users SET is_active = true WHERE id = %s", (target_user_id,))
    log_audit_action(current_user["user_id"], target_user_id, "reactivate_user", {})


def update_managed_user(
    current_user: dict,
    target_user_id: str,
    role: Optional[str] = None,
    email: Optional[str] = None,
) -> dict:
    """
    Change a managed user's role and/or email.

    Previously the only way to promote someone to manager_admin, or to correct a
    typo'd email, was direct SQL. Role changes are restricted to super admins:
    a manager_admin may edit the employees it created, but may not change
    anyone's role, which would otherwise let it mint peers.
    """
    target = _verify_managed_user_target(current_user, target_user_id)
    curr_role = current_user.get("role")

    updates: dict[str, object] = {}
    details: dict[str, object] = {}

    if role is not None and role != target["role"]:
        if curr_role != "super_admin":
            raise HTTPException(status_code=403, detail="Only a super admin can change roles.")
        if role not in ("employee", "manager_admin"):
            raise HTTPException(status_code=400, detail="Role must be employee or manager_admin.")
        updates["role"] = role
        details["role"] = {"from": target["role"], "to": role}

    if email is not None:
        clean_email = email.strip() or None
        updates["email"] = clean_email
        details["email"] = clean_email

    if not updates:
        raise HTTPException(status_code=400, detail="Nothing to update.")

    set_sql = ", ".join(f"{k} = %s" for k in updates)
    params = tuple(updates.values()) + (target_user_id,)
    try:
        execute(f"UPDATE users SET {set_sql} WHERE id = %s", params)
    except Exception as e:
        if "unique constraint" in str(e).lower():
            raise HTTPException(status_code=400, detail="That email is already in use.")
        raise HTTPException(status_code=500, detail="Database error updating user.")

    log_audit_action(current_user["user_id"], target_user_id, "update_user", details)

    rows = run_query(
        "SELECT id, username, email, role, is_active FROM users WHERE id = %s",
        (target_user_id,),
    )
    return rows[0] if rows else {}

