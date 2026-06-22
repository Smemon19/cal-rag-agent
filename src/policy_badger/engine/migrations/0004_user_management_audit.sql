-- Migration: User Management Audit
-- Sets up an audit log table to track actions like create_user, reset_password, deactivate_user, reactivate_user, create_company

CREATE TABLE IF NOT EXISTS user_audit_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    actor_user_id UUID REFERENCES users(id),
    target_user_id UUID REFERENCES users(id),
    action TEXT NOT NULL,
    details JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Indexes for frequent queries
CREATE INDEX IF NOT EXISTS idx_user_audit_log_actor_user_id ON user_audit_log(actor_user_id);
CREATE INDEX IF NOT EXISTS idx_user_audit_log_target_user_id ON user_audit_log(target_user_id);
CREATE INDEX IF NOT EXISTS idx_user_audit_log_action ON user_audit_log(action);
