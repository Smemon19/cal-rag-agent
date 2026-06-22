#!/usr/bin/env python3
import sys
import os

# Append src to path so we can import policy_badger.
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, "src"))

from policy_badger.engine.db import run_query

def main():
    print("Checking database schema for Auth (Phase 3)...\n")
    
    passed = True

    try:
        # 1. Check companies table
        rows = run_query("SELECT column_name FROM information_schema.columns WHERE table_name = 'companies'")
        if rows:
            print("[PASS] Table 'companies' exists.")
        else:
            print("[FAIL] Table 'companies' is missing.")
            passed = False
            
        # 2. Check users table
        rows = run_query("SELECT column_name FROM information_schema.columns WHERE table_name = 'users'")
        if rows:
            print("[PASS] Table 'users' exists.")
        else:
            print("[FAIL] Table 'users' is missing.")
            passed = False
            
        # 3. Check user_audit_log table
        rows = run_query("SELECT column_name FROM information_schema.columns WHERE table_name = 'user_audit_log'")
        if rows:
            print("[PASS] Table 'user_audit_log' exists.")
        else:
            print("[FAIL] Table 'user_audit_log' is missing.")
            passed = False
            
        # 4. Check active super admin
        rows = run_query("SELECT count(*) as count FROM users WHERE role = 'super_admin' AND is_active = true")
        count = rows[0]["count"] if rows else 0
        if count > 0:
            print(f"[PASS] Found {count} active super_admin(s).")
        else:
            print("[FAIL] No active super_admin found. You may be locked out!")
            passed = False
            
        # 5. Check no null password hashes
        rows = run_query("SELECT count(*) as count FROM users WHERE password_hash IS NULL")
        null_count = rows[0]["count"] if rows else 0
        if null_count == 0:
            print("[PASS] No NULL password hashes found.")
        else:
            print(f"[FAIL] Found {null_count} users with NULL password hashes.")
            passed = False
            
        # 6. Check constraint
        # Not easily querying the precise constraint name across postgres versions without complexity,
        # but we can try to find any CHECK constraint on the users table.
        rows = run_query("""
            SELECT conname 
            FROM pg_constraint 
            WHERE conrelid = 'users'::regclass AND contype = 'c'
        """)
        if rows:
            print(f"[PASS] Found check constraint(s) on users table: {rows[0]['conname']}")
        else:
            print("[FAIL] No CHECK constraints found on users table (roles may not be enforced safely).")
            passed = False

    except Exception as e:
        print(f"[ERROR] Database connection or query failed: {e}")
        passed = False
        
    print("\nSummary:")
    if passed:
        print("[OK] Schema validation passed.")
        sys.exit(0)
    else:
        print("[FAIL] Schema validation failed. Check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
