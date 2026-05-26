# Phase 3: Production Deployment & Hardening Checklist

This document details the exact steps necessary to securely deploy the Phase 3 RBAC and Authentication system to Google Cloud Run and Cloud SQL.

## 1. Database Migrations

Before deploying the Cloud Run service, ensure the schema is up to date in your production Cloud SQL instance.

1. Connect to your Cloud SQL instance using the Cloud SQL Auth Proxy.
2. Apply the foundational auth migration:
   ```bash
   psql -h 127.0.0.1 -d policy-badger -U postgres -f policy_engine/migrations/0003_auth_and_roles_foundation.sql
   ```
3. Apply the audit log migration:
   ```bash
   psql -h 127.0.0.1 -d policy-badger -U postgres -f policy_engine/migrations/0004_user_management_audit.sql
   ```

## 2. Set Up Secret Manager

Do not hardcode the initial Super Admin password in scripts or environment variables. Instead, use Google Secret Manager.

1. Create a secret for the seed password:
   ```bash
   echo -n "YourSuperSecurePassword123!" | gcloud secrets create SEED_SUPER_ADMIN_PASSWORD --data-file=-
   ```
2. Verify existing secrets (`DB_HOST`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`, `APP_SECRET_KEY`) exist and are accessible by your Cloud Run Service Account.

## 3. Deployment Command

Run the updated `deploy.sh` script. The script automatically passes `ENVIRONMENT=production` and `ENABLE_LEGACY_APP_USERS_AUTH=false`, fully locking out legacy configuration fallbacks.

```bash
# If you want to seed a specific username instead of reading from .env locally, you can pass it if modified in script or .env
./deploy/deploy.sh
```

*(Note: In `deploy.sh`, `SEED_SUPER_ADMIN_PASSWORD` is safely mapped from Secret Manager)*

## 4. Post-Deployment Verification

### A. Verify Database Schema
From your local machine (connected to the auth proxy), run the schema validation script to ensure no corrupted constraints or missing tables:
```bash
DB_HOST=127.0.0.1 DB_NAME=policy-badger DB_USER=postgres DB_PASSWORD=your_pass python scripts/check_auth_schema.py
```

### B. Run Production Smoke Tests
Run the provided Python smoke script against the newly deployed Cloud Run URL:
```bash
BASE_URL="https://your-cloud-run-url.a.run.app" \
SMOKE_USERNAME="your-seeded-admin-username" \
SMOKE_PASSWORD="YourSuperSecurePassword123!" \
python scripts/smoke_test_auth.py
```
This script safely checks that the session layer, login redirects, and RBAC `/me` endpoints operate without explicitly sending potentially destructive prompts or scraping hashes.

### C. Manual Verification
- Navigate to `https://your-cloud-run-url.a.run.app/login` in an incognito window.
- Log in with the newly seeded super admin.
- Confirm you are prompted to reset your password if `must_reset_password` was applied, or reset it manually anyway.
- Ensure the `/admin` interface loads properly.

## 5. Clean Up

**CRITICAL:** Once the initial Super Admin has logged in and rotated their password via the UI or `Change Password` screen, it is highly recommended to disable or destroy the `SEED_SUPER_ADMIN_PASSWORD` secret in Secret Manager to prevent accidental resets during future deployments. 
```bash
gcloud secrets versions disable latest --secret="SEED_SUPER_ADMIN_PASSWORD"
```

The system will no longer attempt to seed the user if the user already exists in the database.
