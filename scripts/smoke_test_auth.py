#!/usr/bin/env python3
import os
import sys
import requests

def main():
    base_url = os.environ.get("BASE_URL")
    username = os.environ.get("SMOKE_USERNAME")
    password = os.environ.get("SMOKE_PASSWORD")

    if not base_url:
        print("[FAIL] BASE_URL environment variable is required.")
        sys.exit(1)
        
    base_url = base_url.rstrip("/")
    print(f"Running smoke tests against: {base_url}")

    session = requests.Session()
    
    # 1. Test GET /login
    print("Testing GET /login...")
    try:
        res = session.get(f"{base_url}/login", timeout=5)
        if res.status_code == 200:
            print("[PASS] GET /login")
        else:
            print(f"[FAIL] GET /login returned {res.status_code}")
            sys.exit(1)
    except Exception as e:
        print(f"[FAIL] GET /login exception: {e}")
        sys.exit(1)

    if not username or not password:
        print("\n[INFO] No SMOKE_USERNAME or SMOKE_PASSWORD provided. Skipping authenticated tests.")
        sys.exit(0)

    # 2. Test POST /login
    print("\nTesting POST /login...")
    try:
        res = session.post(f"{base_url}/login", data={"username": username, "password": password}, allow_redirects=False, timeout=5)
        if res.status_code in [302, 303]:
            # Expecting redirect to / or /change-password
            print(f"[PASS] POST /login redirect to {res.headers.get('location')}")
            
            # Follow it
            res = session.get(f"{base_url}{res.headers.get('location')}", timeout=5)
            if res.status_code == 200:
                print(f"[PASS] Successfully reached {res.url} after login.")
            else:
                print(f"[FAIL] Failed to load redirected page, got {res.status_code}")
                sys.exit(1)
        else:
            print(f"[FAIL] POST /login returned {res.status_code}. Auth failed.")
            sys.exit(1)
    except Exception as e:
        print(f"[FAIL] POST /login exception: {e}")
        sys.exit(1)

    # 3. Test /api/auth/me
    print("\nTesting GET /api/auth/me...")
    role = "employee" # default assumption
    try:
        res = session.get(f"{base_url}/api/auth/me", timeout=5)
        if res.status_code == 200:
            data = res.json()
            if data.get("username") == username:
                role = data.get("role", "employee")
                print(f"[PASS] GET /api/auth/me returned username matching and role '{role}'")
            else:
                print(f"[FAIL] GET /api/auth/me username mismatch (got {data.get('username')})")
                sys.exit(1)
        else:
            print(f"[FAIL] GET /api/auth/me returned {res.status_code}")
            sys.exit(1)
    except Exception as e:
        print(f"[FAIL] GET /api/auth/me exception: {e}")
        sys.exit(1)

    # 4. Test /api/admin/users
    if role in ["manager_admin", "super_admin"]:
        print("\nTesting GET /api/admin/users...")
        try:
            res = session.get(f"{base_url}/api/admin/users", timeout=5)
            if res.status_code == 200:
                print("[PASS] GET /api/admin/users")
            else:
                print(f"[FAIL] GET /api/admin/users returned {res.status_code}")
                sys.exit(1)
        except Exception as e:
            print(f"[FAIL] GET /api/admin/users exception: {e}")
            sys.exit(1)

    # 5. Skip /ask safely
    print("\n[INFO] Skipping /ask POST safely to avoid accidental prompt injections.")
    
    print("\n[OK] All smoke tests passed successfully.")
    sys.exit(0)

if __name__ == "__main__":
    main()
