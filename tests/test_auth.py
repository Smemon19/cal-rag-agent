import pytest
import bcrypt
from policy_badger.engine.auth_db import (
    verify_user_password,
    get_user_by_username,
    authenticate_user
)

def test_verify_user_password():
    plain = "SecurePass123!"
    hashed = bcrypt.hashpw(plain.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    # Correct password
    assert verify_user_password(plain, hashed) is True
    
    # Incorrect password
    assert verify_user_password("WrongPass123!", hashed) is False

def test_authenticate_user_success(monkeypatch):
    plain = "MySecret"
    hashed = bcrypt.hashpw(plain.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    def mock_get_user(username):
        return {
            "id": "123",
            "username": username,
            "password_hash": hashed,
            "role": "employee",
            "company_id": "456",
            "is_active": True
        }
        
    monkeypatch.setattr("policy_badger.engine.auth_db.get_user_by_username", mock_get_user)
    
    user = authenticate_user("testuser", plain)
    assert user is not None
    assert user["username"] == "testuser"
    assert user["role"] == "employee"

def test_authenticate_user_wrong_password(monkeypatch):
    plain = "MySecret"
    hashed = bcrypt.hashpw(plain.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    def mock_get_user(username):
        return {
            "id": "123",
            "username": username,
            "password_hash": hashed,
            "role": "employee",
            "company_id": "456",
            "is_active": True
        }
        
    monkeypatch.setattr("policy_badger.engine.auth_db.get_user_by_username", mock_get_user)
    
    user = authenticate_user("testuser", "WrongSecret")
    assert user is None

def test_authenticate_user_not_found(monkeypatch):
    def mock_get_user(username):
        return None
        
    monkeypatch.setattr("policy_badger.engine.auth_db.get_user_by_username", mock_get_user)
    
    user = authenticate_user("missinguser", "Password")
    assert user is None

def test_manager_cannot_create_manager(monkeypatch):
    from policy_badger.engine.auth_db import create_managed_user
    from fastapi import HTTPException
    
    current_user = {
        "user_id": "1",
        "role": "manager_admin",
        "company_id": "c1"
    }
    
    with pytest.raises(HTTPException) as exc:
        create_managed_user(current_user, "new_manager", "pass", "manager_admin")
    assert exc.value.status_code == 403
    assert "only create employee" in exc.value.detail.lower()

def test_super_admin_can_create_manager(monkeypatch):
    from policy_badger.engine.auth_db import create_managed_user
    
    current_user = {
        "user_id": "1",
        "role": "super_admin"
    }
    
    def mock_create(*args, **kwargs):
        return "new-uuid"
        
    def mock_log(*args, **kwargs):
        pass
        
    monkeypatch.setattr("policy_badger.engine.auth_db.create_user", mock_create)
    monkeypatch.setattr("policy_badger.engine.auth_db._get_raymond_global_company_id", lambda: "company-1")
    monkeypatch.setattr("policy_badger.engine.auth_db.log_audit_action", mock_log)
    
    res = create_managed_user(current_user, "new_manager", "pass", "manager_admin")
    assert res["username"] == "new_manager"
    assert res["role"] == "manager_admin"

def test_manager_cannot_deactivate_outside_company(monkeypatch):
    from policy_badger.engine.auth_db import deactivate_managed_user
    from fastapi import HTTPException
    
    current_user = {
        "user_id": "1",
        "role": "manager_admin",
        "company_id": "c1"
    }
    
    def mock_run_query(sql, params):
        return [{"id": "2", "role": "employee", "company_id": "c2"}]
        
    monkeypatch.setattr("policy_badger.engine.auth_db.run_query", mock_run_query)
    
    with pytest.raises(HTTPException) as exc:
        deactivate_managed_user(current_user, "2")
    assert exc.value.status_code == 403
    assert "not authorized" in exc.value.detail.lower()

def test_super_admin_cannot_deactivate_super_admin(monkeypatch):
    from policy_badger.engine.auth_db import deactivate_managed_user
    from fastapi import HTTPException
    
    current_user = {
        "user_id": "1",
        "role": "super_admin",
    }
    
    def mock_run_query(sql, params):
        return [{"id": "2", "role": "super_admin", "company_id": "c1"}]
        
    monkeypatch.setattr("policy_badger.engine.auth_db.run_query", mock_run_query)
    
    with pytest.raises(HTTPException) as exc:
        deactivate_managed_user(current_user, "2")
    assert exc.value.status_code == 403
def test_manager_cannot_reactivate_manager_same_company(monkeypatch):
    from policy_badger.engine.auth_db import reactivate_managed_user
    from fastapi import HTTPException
    
    current_user = {"user_id": "1", "role": "manager_admin", "company_id": "c1"}
    
    def mock_run_query(sql, params):
        return [{"id": "2", "role": "manager_admin", "company_id": "c1"}]
        
    monkeypatch.setattr("policy_badger.engine.auth_db.run_query", mock_run_query)
    
    with pytest.raises(HTTPException) as exc:
        reactivate_managed_user(current_user, "2")
    assert exc.value.status_code == 403
    assert "not authorized" in exc.value.detail.lower()

def test_manager_cannot_reset_manager_same_company(monkeypatch):
    from policy_badger.engine.auth_db import reset_managed_user_password
    from fastapi import HTTPException
    
    current_user = {"user_id": "1", "role": "manager_admin", "company_id": "c1"}
    
    def mock_run_query(sql, params):
        return [{"id": "2", "role": "manager_admin", "company_id": "c1"}]
        
    monkeypatch.setattr("policy_badger.engine.auth_db.run_query", mock_run_query)
    
    with pytest.raises(HTTPException) as exc:
        reset_managed_user_password(current_user, "2", "newpass")
    assert exc.value.status_code == 403
    assert "not authorized" in exc.value.detail.lower()

def test_manager_cannot_deactivate_manager_same_company(monkeypatch):
    from policy_badger.engine.auth_db import deactivate_managed_user
    from fastapi import HTTPException
    
    current_user = {"user_id": "1", "role": "manager_admin", "company_id": "c1"}
    
    def mock_run_query(sql, params):
        return [{"id": "2", "role": "manager_admin", "company_id": "c1"}]
        
    monkeypatch.setattr("policy_badger.engine.auth_db.run_query", mock_run_query)
    
    with pytest.raises(HTTPException) as exc:
        deactivate_managed_user(current_user, "2")
    assert exc.value.status_code == 403
    assert "not authorized" in exc.value.detail.lower()

def test_super_admin_can_reactivate_manager(monkeypatch):
    from policy_badger.engine.auth_db import reactivate_managed_user
    
    current_user = {"user_id": "1", "role": "super_admin"}
    
    def mock_run_query(sql, params):
        return [{"id": "2", "role": "manager_admin", "company_id": "c1"}]
        
    def mock_execute(sql, params):
        pass
        
    def mock_log(*args, **kwargs):
        pass
        
    monkeypatch.setattr("policy_badger.engine.auth_db.run_query", mock_run_query)
    monkeypatch.setattr("policy_badger.engine.auth_db.execute", mock_execute)
    monkeypatch.setattr("policy_badger.engine.auth_db.log_audit_action", mock_log)
    
    # Should not raise
    reactivate_managed_user(current_user, "2")

def test_employee_gets_403_on_get_users():
    from policy_badger.engine.auth_db import require_roles
    from fastapi import HTTPException
    
    class FakeRequest:
        session = {"user_id": "1", "username": "emp", "role": "employee"}
    
    with pytest.raises(HTTPException) as exc:
        require_roles(FakeRequest(), {"manager_admin", "super_admin"})
    assert exc.value.status_code == 403

def test_password_hash_not_returned_from_list_users(monkeypatch):
    from policy_badger.engine.auth_db import list_users_for_admin
    
    current_user = {"user_id": "1", "role": "super_admin"}
    
    def mock_run_query(sql, params=None):
        return [{"id": "2", "username": "u", "email": "e", "role": "employee", "is_active": True, "company_id": "c1", "company_name": "C"}]
        
    monkeypatch.setattr("policy_badger.engine.auth_db.run_query", mock_run_query)
    
    res = list_users_for_admin(current_user)
    assert len(res) == 1
    assert "password_hash" not in res[0]

def test_production_legacy_auth_guard():
    import subprocess
    import sys
    import os
    
    # We test this by trying to run `python -c 'import policy_badger.web.app'` with the bad env vars
    # It should sys.exit(1) due to the safety guard.
    env = os.environ.copy()
    env["ENVIRONMENT"] = "production"
    env["ENABLE_LEGACY_APP_USERS_AUTH"] = "true"
    env["ALLOW_PROD_LEGACY_AUTH"] = "false"
    env["PYTHONPATH"] = os.path.join(os.getcwd(), "src")
    
    # We must be in the correct directory so imports work
    cwd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    result = subprocess.run(
        [sys.executable, "-c", "import policy_badger.web.app"],
        env=env,
        cwd=cwd,
        capture_output=True,
        text=True
    )
    
    assert result.returncode == 1
    assert "CRITICAL: ENABLE_LEGACY_APP_USERS_AUTH is true in production" in result.stderr
