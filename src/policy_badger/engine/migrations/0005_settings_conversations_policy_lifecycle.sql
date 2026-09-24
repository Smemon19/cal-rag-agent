-- Migration: Per-user settings, conversation history, and policy lifecycle.
--
-- Covers three previously unbuilt areas:
--   1. user_settings         — answer preferences, replacing browser-only localStorage
--   2. conversations/messages — durable chat history behind the sidebar rail
--   3. policies_v2 lifecycle  — edit / supersede / archive instead of publish-only
--
-- Safe to re-run: every statement is IF NOT EXISTS or guarded by a DO block.

-- Defined by 0003; repeated here so this file can be applied on its own.
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- ── 1. Per-user answer settings ──────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS user_settings (
    user_id UUID PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    settings JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'set_user_settings_updated_at') THEN
        CREATE TRIGGER set_user_settings_updated_at
            BEFORE UPDATE ON user_settings
            FOR EACH ROW
            EXECUTE FUNCTION update_updated_at_column();
    END IF;
END $$;

-- ── 2. Conversation history ──────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title TEXT NOT NULL DEFAULT 'New conversation',
    is_archived BOOLEAN NOT NULL DEFAULT false,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'set_conversations_updated_at') THEN
        CREATE TRIGGER set_conversations_updated_at
            BEFORE UPDATE ON conversations
            FOR EACH ROW
            EXECUTE FUNCTION update_updated_at_column();
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_conversations_user_updated
    ON conversations(user_id, updated_at DESC);

CREATE TABLE IF NOT EXISTS conversation_messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    rows_found INT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT valid_message_role CHECK (role IN ('user', 'assistant'))
);

CREATE INDEX IF NOT EXISTS idx_conversation_messages_conversation
    ON conversation_messages(conversation_id, created_at);

-- ── 3. policies_v2 lifecycle columns ─────────────────────────────────────────
-- policies_v2 is created by the adaptive-ingestion schema evolution path rather
-- than by a numbered migration, so only add columns if the table is present.
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.tables
        WHERE table_schema = 'public' AND table_name = 'policies_v2'
    ) THEN
        ALTER TABLE policies_v2 ADD COLUMN IF NOT EXISTS status TEXT NOT NULL DEFAULT 'active';
        ALTER TABLE policies_v2 ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT now();
        ALTER TABLE policies_v2 ADD COLUMN IF NOT EXISTS superseded_by_policy_id TEXT;
        ALTER TABLE policies_v2 ADD COLUMN IF NOT EXISTS archived_at TIMESTAMPTZ;
        ALTER TABLE policies_v2 ADD COLUMN IF NOT EXISTS archived_by_user_id UUID REFERENCES users(id);

        CREATE INDEX IF NOT EXISTS idx_policies_v2_status ON policies_v2(status);

        IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'set_policies_v2_updated_at') THEN
            CREATE TRIGGER set_policies_v2_updated_at
                BEFORE UPDATE ON policies_v2
                FOR EACH ROW
                EXECUTE FUNCTION update_updated_at_column();
        END IF;
    END IF;
END $$;

-- ── 4. Policy administration audit ───────────────────────────────────────────
-- admin_policy_publish_audit records publishes only. Edits, archives and
-- restores need their own trail.
CREATE TABLE IF NOT EXISTS policy_admin_audit (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    actor_user_id UUID REFERENCES users(id),
    policy_id TEXT NOT NULL,
    action TEXT NOT NULL,
    details JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_policy_admin_audit_policy ON policy_admin_audit(policy_id);
CREATE INDEX IF NOT EXISTS idx_policy_admin_audit_created ON policy_admin_audit(created_at DESC);
