-- Adaptive ingestion foundation schema.
-- Safe to run multiple times; objects are created if missing.

CREATE TABLE IF NOT EXISTS ingestion_batches (
    batch_id TEXT PRIMARY KEY,
    triggered_by TEXT,
    trigger_source TEXT NOT NULL DEFAULT 'manual',
    status TEXT NOT NULL DEFAULT 'running',
    documents_total INTEGER NOT NULL DEFAULT 0,
    documents_processed INTEGER NOT NULL DEFAULT 0,
    sections_total INTEGER NOT NULL DEFAULT 0,
    extractions_total INTEGER NOT NULL DEFAULT 0,
    errors_total INTEGER NOT NULL DEFAULT 0,
    result_summary TEXT,
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS documents (
    document_id TEXT PRIMARY KEY,
    batch_id TEXT NOT NULL REFERENCES ingestion_batches(batch_id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    doc_type TEXT,
    version TEXT,
    effective_date DATE,
    source_uri TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    raw_content TEXT,
    ingest_status TEXT NOT NULL DEFAULT 'registered',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (source_uri, content_hash)
);

CREATE TABLE IF NOT EXISTS sections (
    section_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    heading TEXT,
    section_text TEXT NOT NULL,
    ordinal INTEGER NOT NULL,
    source_locator TEXT,
    chunk_hash TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS policy_extractions_staging (
    extraction_id TEXT PRIMARY KEY,
    batch_id TEXT NOT NULL REFERENCES ingestion_batches(batch_id) ON DELETE CASCADE,
    -- NOTE: Do not FK to legacy documents/sections tables because
    -- existing deployments may have integer IDs there.
    document_id TEXT NOT NULL,
    section_id TEXT NOT NULL,
    candidate_json JSONB NOT NULL,
    mapped_fields_json JSONB NOT NULL,
    unmapped_concepts_json JSONB NOT NULL,
    confidence DOUBLE PRECISION NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    schema_gap_notes TEXT,
    reviewer_notes TEXT,
    publish_error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS schema_change_requests (
    request_id TEXT PRIMARY KEY,
    batch_id TEXT NOT NULL REFERENCES ingestion_batches(batch_id) ON DELETE CASCADE,
    proposal_json JSONB NOT NULL,
    migration_class TEXT NOT NULL,
    decision_status TEXT NOT NULL DEFAULT 'pending',
    reviewer TEXT,
    decision_note TEXT,
    decided_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS schema_migration_log (
    migration_id TEXT PRIMARY KEY,
    request_id TEXT NOT NULL REFERENCES schema_change_requests(request_id) ON DELETE CASCADE,
    migration_sql_json JSONB NOT NULL,
    applied_by TEXT,
    applied_at TIMESTAMPTZ,
    status TEXT NOT NULL DEFAULT 'pending',
    error_text TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS policy_publish_audit (
    publish_id TEXT PRIMARY KEY,
    extraction_id TEXT NOT NULL REFERENCES policy_extractions_staging(extraction_id) ON DELETE CASCADE,
    policy_id TEXT,
    publish_status TEXT NOT NULL,
    error_text TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS schema_dictionary_overrides (
    override_id TEXT PRIMARY KEY,
    canonical_name TEXT NOT NULL,
    aliases JSONB NOT NULL DEFAULT '[]'::jsonb,
    reserved_names JSONB NOT NULL DEFAULT '[]'::jsonb,
    mapping_rules JSONB NOT NULL DEFAULT '{}'::jsonb,
    active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = 'documents' AND column_name = 'batch_id'
    ) THEN
        EXECUTE 'CREATE INDEX IF NOT EXISTS idx_documents_batch_id ON documents(batch_id)';
    END IF;
END$$;
CREATE INDEX IF NOT EXISTS idx_sections_document_id ON sections(document_id);
CREATE INDEX IF NOT EXISTS idx_staging_batch_status ON policy_extractions_staging(batch_id, status);
CREATE INDEX IF NOT EXISTS idx_schema_requests_batch ON schema_change_requests(batch_id);

