CREATE TABLE IF NOT EXISTS admin_policy_publish_audit (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    submission_id UUID NOT NULL,
    published_policy_id UUID NOT NULL,
    submitted_by VARCHAR(255),
    published_by VARCHAR(255),
    raw_text TEXT NOT NULL,
    extracted_json JSONB,
    final_json JSONB,
    source_type VARCHAR(50) DEFAULT 'admin_entry',
    version INT DEFAULT 1,
    replaces_policy_id UUID,
    created_at TIMESTAMP WITH TIME ZONE,
    published_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
