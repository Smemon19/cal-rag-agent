-- 0006: store policy ids in admin_policy_publish_audit as TEXT
--
-- 0002 declared published_policy_id and replaces_policy_id as UUID, but
-- policies_v2.policy_id is an integer. Every Policy Studio publish therefore
-- failed on the audit insert ("invalid input syntax for type uuid"), and since
-- the policy insert and the audit insert now share one transaction, no Studio
-- publish could land until this runs. TEXT matches policy_publish_audit.policy_id.
--
-- Safe to re-run: the table is created if absent and the type changes are
-- guarded on the current column type.

CREATE TABLE IF NOT EXISTS admin_policy_publish_audit (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    submission_id UUID NOT NULL,
    published_policy_id TEXT NOT NULL,
    submitted_by VARCHAR(255),
    published_by VARCHAR(255),
    raw_text TEXT NOT NULL,
    extracted_json JSONB,
    final_json JSONB,
    source_type VARCHAR(50) DEFAULT 'admin_entry',
    version INT DEFAULT 1,
    replaces_policy_id TEXT,
    created_at TIMESTAMP WITH TIME ZONE,
    published_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = 'admin_policy_publish_audit'
          AND column_name = 'published_policy_id' AND data_type = 'uuid'
    ) THEN
        ALTER TABLE admin_policy_publish_audit
            ALTER COLUMN published_policy_id TYPE TEXT USING published_policy_id::text;
    END IF;

    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = 'admin_policy_publish_audit'
          AND column_name = 'replaces_policy_id' AND data_type = 'uuid'
    ) THEN
        ALTER TABLE admin_policy_publish_audit
            ALTER COLUMN replaces_policy_id TYPE TEXT USING replaces_policy_id::text;
    END IF;
END $$;
