## Cloud Run deployment guide (Cal RAG Agent)

This guide sets sane defaults to minimize cold starts and avoid OOMs. Copy the Golden Deployment Config to get started, then tune using the knobs below.

### Quick deploy (copy-paste)

- Using source (builds from current directory):

```bash
gcloud run deploy cal-rag-agent \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars OPENAI_API_KEY=sk-proj-REPLACE_ME,MODEL_CHOICE=gpt-4.1-mini,EMBEDDING_BACKEND=sentence,HF_HOME=/tmp/hf_cache,TRANSFORMERS_CACHE=/tmp/hf_cache/transformers,SENTENCE_TRANSFORMERS_HOME=/tmp/hf_cache/sentence-transformers,TORCH_HOME=/tmp/hf_cache/torch
  --set-env-vars RAG_COLLECTION_NAME=docs_ibc_v2
```

- Using a prebuilt image:

```bash
PROJECT_ID=$(gcloud config get-value project)
gcloud run deploy cal-rag-agent \
  --image gcr.io/${PROJECT_ID}/cal-rag-agent:latest \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars OPENAI_API_KEY=sk-proj-REPLACE_ME,MODEL_CHOICE=gpt-4.1-mini,EMBEDDING_BACKEND=sentence,HF_HOME=/tmp/hf_cache,TRANSFORMERS_CACHE=/tmp/hf_cache/transformers,SENTENCE_TRANSFORMERS_HOME=/tmp/hf_cache/sentence-transformers,TORCH_HOME=/tmp/hf_cache/torch
  --set-env-vars RAG_COLLECTION_NAME=docs_ibc_v2
```

Notes:

- The service-wide `OPENAI_API_KEY` is used for all users (no keys in the UI).
- File-based secrets are supported via `OPENAI_API_KEY_FILE` if you prefer `--set-secrets`.
- Keep `EMBEDDING_BACKEND=sentence` unless you explicitly want OpenAI embeddings.
- On Cloud Run only `/tmp` is writable; the deploy commands above route Hugging Face caches there.

### Service shape

- Memory: start at 2–4 GiB. If OOMs persist, raise in 1–2 GiB steps.
- CPU: 1–2 vCPU.
- Concurrency: 1 (Streamlit is single-user per instance).
- Request timeout: 180–300 s, aligned to the longest retrieval/LLM round you allow.

### Scaling

- Min instances: 1–2 to suppress cold starts.
- Max instances: set a conservative cap to avoid costly fan-out (e.g., 5–10 for early prod).

### Region

- Choose the region closest to users and your vector DB/LLM endpoints to reduce p99 latency.

### Environment variables (knobs)

Recommended starting values; adjust as needed.

- EMBED_CACHE_SIZE: 2048
  - Bounded LRU cache for embeddings; 0 disables caching.
- EMBED_BATCH_SIZE: 64
  - Batch size for multi-text embedding calls (ingest/query-time). Sequential; retries per batch.
- KEYWORD_PREFILTER_LIMIT: 5000
  - If collection count exceeds this, skip client-side keyword scan and fallback to vector-only.
- KEYWORD_SCAN_LIMIT: 2000
  - Max rows scanned via client-side keyword prefilter when under the prefilter limit.
- VECTOR_CANDIDATE_CAP: 50
  - Upper bound on initial candidates before rerank/filter.
- RETRIEVAL_N_RESULTS: 8
  - Max results returned by vector query for the final context step.
- MAX_CHUNKS_FOR_CONTEXT: 12
  - Cap on chunks included in the formatted context.
- MAX_CONTEXT_TOKENS: 6000
  - Hard limit on context tokens; trips a friendly circuit breaker.
- PER_CHUNK_CHAR_LIMIT: 4000
  - Individual chunk trim size before concatenation to reduce spikes.

Security and keys:

- Inject API keys via environment variables (e.g., OPENAI_API_KEY). Do not bake secrets into images.
- Structured logs do NOT include prompts/documents; they log only lengths/counters.

File system and state:

- Caches are in-process; each instance maintains its own. State resets on restart/scale events.
- Do not rely on local disk persistence across restarts. Chroma runs in Persistent mode against your chosen directory but on Cloud Run only /tmp is writable.

### Troubleshooting

- OOM termination: Logs show container termination and memory spike just before exit. Increase Memory, reduce MAX_CHUNKS_FOR_CONTEXT or MAX_CONTEXT_TOKENS, or raise caps gradually.
- Retry storms: Many overlapping cold starts and throttling. Reduce Max instances, increase Min instances, or increase concurrency only if your app is stateless (not recommended for Streamlit UI).
- Slow starts: Distinguish between image cold start, model warm-up (embeddings), and vector store connectivity. Keep Min instances > 0 and ensure regional proximity.

### Golden Deployment Config (copy-friendly)

Environment variables:

```
EMBED_CACHE_SIZE=2048
EMBED_BATCH_SIZE=64
KEYWORD_PREFILTER_LIMIT=5000
KEYWORD_SCAN_LIMIT=2000
VECTOR_CANDIDATE_CAP=50
RETRIEVAL_N_RESULTS=8
MAX_CHUNKS_FOR_CONTEXT=12
MAX_CONTEXT_TOKENS=6000
PER_CHUNK_CHAR_LIMIT=4000
MODEL_CHOICE=gpt-4.1-mini
# OPENAI_API_KEY=... (set securely)
```

Service settings:

- Region: choose nearest (e.g., us-central1)
- Memory: 4 GiB
- CPU: 2 vCPU
- Concurrency: 1
- Timeout: 240 s
- Min instances: 1
- Max instances: 8

### Health and logs (operations)

The app emits single-line JSON logs per query and exposes a Diagnostics sidebar UI.

Per-query log fields:

- ts, request_id, elapsed_ms, query_chars
- embed_cache_hit (true/false), embed_batches, batch_size
- prefilter_count, prefilter_limit, prefilter_fallback
- n_results_requested, n_results_returned
- candidates_from_vector, candidates_after_cap, deduped_count, trimmed_chunks_count
- chunks_used, tokens_in_context, circuit_breaker_triggered
- errors (null or short code)

PII policy:

- Prompts and documents are not logged verbatim; only lengths and counts are recorded.

How to read the logs:

- prefilter_fallback spikes → queries are too broad; add UI guidance or rely more on vector.
- tokens_in_context near MAX_CONTEXT_TOKENS → reduce caps or encourage narrower queries.
- embed_cache_hit low → increase EMBED_CACHE_SIZE or EMBED_BATCH_SIZE, or review model churn.

### Soak test and rollback (quick plan)

Pre-deploy checks:

- Ensure caps (RETRIEVAL_N_RESULTS, MAX_CHUNKS_FOR_CONTEXT, MAX_CONTEXT_TOKENS, VECTOR_CANDIDATE_CAP, KEYWORD_PREFILTER_LIMIT, KEYWORD_SCAN_LIMIT) appear in logs on the first request.
- Confirm Diagnostics shows reused=true for client/collection on the second query.

Soak scenarios (5–10 minutes each):

- Broad queries to trigger keyword fallback (prefilter_fallback=true); observe no full scans.
- Long-document queries to test context size; confirm no circuit breaker unless abusive.
- Burst test via rapid UI interactions; observe no duplicate workers or re-inits.
- Ingestion batch: embed a medium corpus; verify batch logs, stable peak memory, retry isolation.

Success criteria:

- No OOMs or forced restarts.
- tokens_in_context ≤ MAX_CONTEXT_TOKENS on every run.
- circuit_breaker_triggered false except intentional abuse.
- p95 elapsed steady after warm-up.

Rollback:

- Keep previous container image tagged; rollback is a single redeploy pointing to the prior image.
- Alternatively, revert env changes to prior known-good values.

Post-deploy watch (first hour):

- Monitor prefilter_fallback, embed_cache_hit ratio, and errors.
- If prefilter_fallback > 30% on normal traffic, tighten UI guidance or rely more on vector search.
