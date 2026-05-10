import os
import sys
import psycopg2
from dotenv import load_dotenv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from policy_engine.db import get_connection

def main():
    if len(sys.argv) < 2:
        print("Usage: python apply_migration.py <path_to_sql_file>")
        sys.exit(1)

    sql_file = sys.argv[1]
    if not os.path.exists(sql_file):
        print(f"File not found: {sql_file}")
        sys.exit(1)

    from pathlib import Path
    from dotenv import load_dotenv
    
    ROOT = Path(__file__).resolve().parents[1]
    load_dotenv(ROOT / ".env", override=True)
    print(f"Loaded .env from: {ROOT / '.env'}")
    
    with open(sql_file, 'r') as f:
        sql = f.read()

    print(f"Applying migration: {sql_file}")
    
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
            
            # Verify table exists
            cur.execute("SELECT to_regclass('public.admin_policy_publish_audit');")
            result = cur.fetchone()
            if result and result[0]:
                print(f"Verification successful: table '{result[0]}' exists.")
            else:
                print("Verification failed: table does not exist.")
                conn.rollback()
                sys.exit(1)
                
        conn.commit()
        print("Migration applied successfully.")
    except Exception as e:
        conn.rollback()
        print(f"Failed to apply migration: {e}")
        sys.exit(1)
    finally:
        conn.close()

if __name__ == "__main__":
    main()
