# Cleanup Notes

This branch keeps the current structured Policy Badger system in place and archives files that clearly belonged to the older Chroma/vector RAG or local/generated tooling setup.

No archived files were deleted. They were moved under `legacy_archive/` so they can be restored or referenced later.

## Kept As Current System

These files and folders are treated as the active Policy Badger codebase:

- `app.py`
- `policy_engine/`
- `adaptive_ingestion/`
- `policy_engine/migrations/`
- `adaptive_ingestion/schema_dictionary/`
- `pdf_loader/`
- `public/`
- `scripts/adaptive_ingest.py`
- `scripts/db_status.py`
- `scripts/normalize_topics.py`
- `scripts/manage_users.py`
- `requirements.txt`
- `Dockerfile`
- `apphosting.yaml`
- `firebase.json`
- `.env.example`
- `DEPENDS_NOTES.md`
- `CLOUD_RUN_SETUP.md`
- `OPERATIONS_RUNBOOK.md`
- `tests/test_adaptive_ingestion_units.py`
- `tests/test_adaptive_migration_policy.py`
- `tests/test_schema_dictionary.py`

## Archived Legacy Or Outdated Files

### `legacy_archive/app_configs/`

- `app.yaml`

Reason: the file pointed App Engine at `streamlit_app.py`, which is not part of the active codebase.

### `legacy_archive/docs/`

- `ARCHITECTURE.md`
- `INSTALL_WINDOWS.md`
- `MIGRATION_NOTES.md`
- `PHASE0_CURRENT_STATE.md`
- `pdf_loader_requirements.txt.old`

Reason: these documents described older Chroma, Streamlit, Windows installer, or migration states that do not match the current FastAPI + Postgres Policy Badger system.

### `legacy_archive/eval/`

- `eval/dataset.jsonl`
- `eval/rubric.py`
- `eval/runner.py`

Reason: the eval runner imported missing legacy modules such as `rag_agent` and `utils`.

### `legacy_archive/scripts/`

- `ingest_folder.py`
- `ingest_pdf.py`
- `init_bigquery.py`
- `test_bq_vector.py`
- `test_retrieval.py`
- `verify_bigquery.py`

Reason: these scripts belonged to the older vector/BigQuery RAG path or imported missing legacy modules such as `utils`, `utils_bigquery`, `utils_vectorstore`, and `rag_agent`.

### `legacy_archive/tests/`

- `test_filtering_fix.py`
- `test_policy_badger_pro_retrieval.py`
- `test_rules_and_tables.py`
- `test_vertex_smoke.py`
- `test_verification_skip.py`

Reason: these tests covered legacy RAG behavior or imported missing modules such as `rag_agent`, `policy_badger_pro`, `verify`, `utils`, or `utils_vertex`.

### `legacy_archive/windows/`

- `build_exe.ps1`
- `build_installer.ps1`
- `installer/CalRAGInstaller.iss`

Reason: these files supported the older Windows/Streamlit packaging path.

### `legacy_archive/generated_artifacts/`

- `sfdx/`
- `firebase/`
- `local-bin/cloud-sql-proxy`
- `tmp_chatbot_eval_latest.json`
- `tmp_chatbot_eval_latest_fixed.json`
- `tmp_chatbot_trace_batch_a6143.json`
- `tmp_llm_batch_report.json`

Reason: these are generated, local, binary, or temporary artifacts rather than active application source.

## Import Updates

`adaptive_ingestion/bigquery_mirror.py` previously imported a missing `utils_bigquery` module. It now contains small local BigQuery helper functions for:

- resolving the BigQuery project
- resolving the dataset
- creating a BigQuery client
- inserting JSON rows

This keeps the adaptive ingestion package importable without restoring the old utility module.

## Test Configuration Updates

- Added `pyproject.toml` so pytest ignores `legacy_archive/`.
- Removed the stale `eval` target from `Makefile` because the legacy eval harness was archived.
- Updated extractor unit tests to set `USE_LLM_SEMANTIC_EXTRACTION=0`; this keeps tests deterministic and avoids live Vertex/Gemini imports during unit tests.

## Dependency Updates

`requirements.txt` now includes dependencies used by retained current modules:

- `google-cloud-bigquery`
- `PyYAML`
- `streamlit`
- `google-cloud-secret-manager`

## Needs Review

These files were left in place because they may still be useful, but their long-term role should be confirmed:

- `policy_engine/ui_app.py` and `policy_engine/schema_admin_ui.py`: optional Streamlit admin/dev UIs for the current policy system.
- `scripts/manage_users.py`: Secret Manager helper for app users.
- `scripts/normalize_topics.py`: likely policy-data cleanup helper, but should be reviewed against the current DB schema.
- `CLOUD_RUN_SETUP.md`, `OPERATIONS_RUNBOOK.md`, and `deploy/deploy.sh`: may still contain useful deployment notes, but may include older vector/RAG references.
- `rules/`, `config/`, `docs/manual/`, and `docs/sample_token_bucket.md`: no active imports found, but they may be sample source data or policy reference material.
- `firebase.json`: static hosting config remains, but it does not currently define a Cloud Run rewrite.

## Checks Run

After archiving files:

```bash
python -m compileall -q app.py policy_engine adaptive_ingestion pdf_loader scripts tests
```

Result: passed.

Initial `pytest -q` after archiving:

```text
legacy_archive/ tests and scripts were still collected by pytest and failed on expected missing legacy modules.
```

Action: added `pyproject.toml` with `norecursedirs = ["legacy_archive", ".git", ".venv", "venv", ".venv_local"]`.

Second `pytest -q` run:

```text
pytest started active tests but hit a Python/numpy segmentation fault while the extractor test attempted the live LLM path.
```

Action: updated extractor unit tests to use deterministic extraction via `USE_LLM_SEMANTIC_EXTRACTION=0`.

Final check:

```bash
python -m compileall -q app.py policy_engine adaptive_ingestion pdf_loader scripts tests
python - <<'PY'
import app
import policy_engine.service
import adaptive_ingestion.pipeline
import adaptive_ingestion.bigquery_mirror
print('core imports ok')
PY
pytest -q
```

Result: `10 passed`.

After fixing the BigQuery mirror import:

```bash
python -m compileall -q app.py policy_engine adaptive_ingestion pdf_loader scripts tests
python - <<'PY'
import app
import policy_engine.service
import adaptive_ingestion.pipeline
import adaptive_ingestion.bigquery_mirror
print('core imports ok')
PY
```

Result: passed.
