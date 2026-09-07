# Policy Badger - Raymond

Policy Badger - Raymond is a structured policy question-answering app backed by PostgreSQL. It answers user questions by planning a constrained database search, building safe SQL, retrieving rows from `policies_v2`, and formatting a grounded answer from those rows.

The current system is centered on the `src/policy_badger/engine/` and `src/policy_badger/ingestion/` packages.

## Core Flow

```text
User question
  -> planner JSON search spec
  -> allowlisted SQL query builder
  -> PostgreSQL policies_v2
  -> formatter answer grounded in returned rows
```

Main implementation:

- `src/policy_badger/web/app.py` is the authenticated FastAPI web app.
- `src/policy_badger/engine/service.py` orchestrates question answering.
- `src/policy_badger/engine/planner.py` asks the LLM for a JSON search spec only.
- `src/policy_badger/engine/query_builder.py` builds parameterized SQL from validated fields and filters.
- `src/policy_badger/engine/db.py` connects to PostgreSQL using `DB_*` environment variables.
- `src/policy_badger/engine/formatter.py` writes the final answer using only retrieved rows.
- `src/policy_badger/ingestion/` ingests policy documents, stages extracted policy candidates, proposes schema changes, and publishes accepted rows.

## Repository Structure

```text
.
├── apphosting.yaml
├── Dockerfile
├── requirements.txt
├── src/
│   └── policy_badger/
│       ├── engine/
│       │   └── migrations/
│       ├── ingestion/
│       ├── pdf/
│       └── web/
│           └── static/
├── scripts/
└── tests/
```

## Web App

Run locally:

```bash
PYTHONPATH=src uvicorn policy_badger.web.app:app --reload
```

The FastAPI app provides:

- `GET /login` and `POST /login` for session login.
- `GET /logout` to clear the session.
- `GET /` for the web UI in `src/policy_badger/web/static/index.html`.
- `POST /ask` for authenticated policy questions.
- `GET /health` for health checks.

Users are configured with either:

- `APP_USERS`, a JSON object like `{"alice": "password1", "bob": "password2"}`.
- Or legacy single-user fallback values `APP_USER` and `APP_PASSWORD`.

Set `APP_SECRET_KEY` in deployed environments.

## Policy Engine

The policy engine is intentionally split into small responsibilities:

- `planner.py` produces a validated JSON search specification. It does not answer questions and does not execute SQL.
- `query_builder.py` converts the spec into safe, parameterized PostgreSQL queries using allowlisted fields only.
- `db.py` opens PostgreSQL connections with `psycopg2`.
- `service.py` coordinates planning, querying, relaxed fallback queries, text fallback queries, and final formatting.
- `formatter.py` asks the LLM to answer using only retrieved rows.

The main public API is:

```python
from policy_badger.engine.service import answer_policy_question

result = answer_policy_question("What is the PTO approval policy?")
```

## Adaptive Ingestion

The adaptive ingestion system turns source policy documents into structured policy rows.

High-level flow:

```text
Document input
  -> ingestion batch
  -> document registration
  -> section chunking
  -> semantic policy extraction
  -> staging rows
  -> schema gap detection
  -> migration proposals
  -> approved migrations
  -> publish into policies_v2
```

Important files:

- `src/policy_badger/ingestion/pipeline.py` wires the ingestion workflow.
- `src/policy_badger/ingestion/policy_extractor.py` extracts structured candidate policy JSON.
- `src/policy_badger/ingestion/schema_dictionary.py` loads canonical policy field definitions.
- `src/policy_badger/ingestion/gap_detector.py` and `schema_planner.py` identify schema gaps.
- `src/policy_badger/ingestion/migration_generator.py` and `migration_applier.py` generate and apply approved schema changes.
- `src/policy_badger/ingestion/publisher.py` validates staged records and publishes them to `policies_v2`.
- `src/policy_badger/engine/migrations/0001_adaptive_ingestion_foundation.sql` creates the ingestion support tables.

Run adaptive ingestion from the CLI:

```bash
python scripts/adaptive_ingest.py path/to/policy.docx
```

Inspect database status:

```bash
python scripts/db_status.py
```

## Configuration

Required for the web app and policy engine:

```env
APP_SECRET_KEY=change-me
APP_USER=Test
APP_PASSWORD=Test123!
OPENAI_API_KEY=...
DB_HOST=...
DB_PORT=5432
DB_NAME=cal_policy_db
DB_USER=postgres
DB_PASSWORD=...
```

Optional for adaptive extraction and mirroring:

```env
USE_LLM_SEMANTIC_EXTRACTION=1
SEMANTIC_EXTRACT_MODEL=gemini-2.0-flash
VERTEX_PROJECT_ID=...
VERTEX_LOCATION=us-central1
BQ_POLICY_DATASET=...
BQ_POLICY_EVENTS_TABLE=policy_events
BQ_POLICY_CURRENT_TABLE=policies_current
```

## Deployment

The current deployment path is Docker/FastAPI:

```bash
docker build -t policy-badger .
docker run --env-file .env -p 8080:8080 policy-badger
```

The `Dockerfile` runs:

```bash
uvicorn policy_badger.web.app:app --host 0.0.0.0 --port ${PORT:-8080}
```

`apphosting.yaml` contains the App Hosting / Cloud Run-style runtime and secret configuration.

## Tests And Checks

Run active tests:

```bash
pytest -q
```

Compile active Python modules:

```bash
python -m compileall -q src scripts tests
```
