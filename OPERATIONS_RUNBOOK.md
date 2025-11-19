Operations Runbook (Firebase App Hosting)

Monitoring (lightweight)

- Enable request/error logs in Firebase Console → App Hosting → Backends → default.
- Add an external uptime check (UptimeRobot/Pingdom) for the hosted URL root. Target: HTTP 200 in <2s median, sample every 5 minutes.
- Configure alerts to team email/SMS in your monitor provider.

Smoke Test

- Open hosted URL, expand Diagnostics sidebar:
  - Confirm masked OPENAI prefix, model, backend, collection, fingerprint, model home `/app/models`.
- Run the slate-roof underlayment query twice on two devices; expect IBC citations and collection `docs_ibc_v2`.

Triage (on issues)

- Check last rollout build logs for baked-model and health pass.
- Verify Diagnostics reflect App Hosting env; ensure OPENAI secret latest version is Enabled.
- Ensure no plain `OPENAI_API_KEY` env var exists.
- If retrieval empty/off-topic: confirm `RAG_COLLECTION_NAME` and check mixed-embedding warnings; switch collection or re-embed.

Rollback

- Firebase Console → App Hosting → Rollouts → select last known good rollout → Promote/Restore to live.
- After rollback, re-verify Diagnostics and the slate-roof query from two devices.

## BigQuery Vector Search Operations

### Backend Configuration

The application supports two vector storage backends:
- **ChromaDB** (default): Local vector storage
- **BigQuery**: Cloud-based vector search

Switch backends via the `VECTOR_BACKEND` environment variable:
```bash
VECTOR_BACKEND=bigquery  # or "chroma"
```

### BigQuery Setup Verification

Run the verification script to check BigQuery configuration:
```bash
python scripts/verify_bigquery.py
```

Expected output:
- ✓ BigQuery client initialized
- ✓ Table exists: cal-rag-agent.calrag.documents
- ✓ Row count: N documents
- ✓ Schema includes: chunk_id, content, embedding, source_url, metadata
- ✓ Vector index status (if created)

### Smoke Test (BigQuery Backend)

1. Set environment:
   ```bash
   export VECTOR_BACKEND=bigquery
   export BQ_PROJECT=cal-rag-agent
   export BQ_DATASET=calrag
   export BQ_TABLE=documents
   ```

2. Open Streamlit UI, check Diagnostics sidebar:
   - Vector backend: bigquery
   - Project: cal-rag-agent
   - Dataset: calrag
   - Document count: matches expected

3. Run test query:
   ```bash
   python scripts/test_bq_vector.py
   ```

### Monitoring BigQuery

#### Query Performance
- **Location**: Cloud Console → BigQuery → Query History
- **Metrics to monitor**:
  - Bytes processed per query
  - Execution time (should be <2s for most queries)
  - Error rate
  - Query costs

#### Cost Tracking
- Set up budget alerts in Cloud Console
- Recommended: Alert at 80% of monthly budget
- Track metrics:
  - Queries per day
  - Average bytes processed
  - Storage costs

#### Performance Tuning
If queries are slow (>2s):
1. Check vector index status:
   ```sql
   SELECT * FROM `cal-rag-agent.calrag.INFORMATION_SCHEMA.VECTOR_INDEXES`
   WHERE table_name = 'documents';
   ```

2. Verify index coverage is 100%
3. If no index exists, create one:
   ```sql
   CREATE VECTOR INDEX documents_embedding_idx
   ON `cal-rag-agent.calrag.documents`(embedding)
   OPTIONS(distance_type='COSINE', index_type='IVF');
   ```

### Troubleshooting BigQuery Issues

#### Error: "Table not found"
**Symptoms**: Application logs show table not found errors

**Resolution**:
1. Verify table exists:
   ```bash
   bq show cal-rag-agent:calrag.documents
   ```
2. Check BQ_PROJECT, BQ_DATASET, BQ_TABLE env vars match actual table
3. Verify service account has `bigquery.tables.get` permission

#### Error: "Permission denied"
**Symptoms**: 403 errors in application logs

**Resolution**:
1. Check service account permissions:
   ```bash
   gcloud projects get-iam-policy cal-rag-agent \
     --flatten="bindings[].members" \
     --filter="bindings.members:<SERVICE_ACCOUNT_EMAIL>"
   ```
2. Required roles:
   - `roles/bigquery.dataViewer`
   - `roles/bigquery.jobUser`

3. Grant missing permissions:
   ```bash
   gcloud projects add-iam-policy-binding cal-rag-agent \
     --member="serviceAccount:<SERVICE_ACCOUNT_EMAIL>" \
     --role="roles/bigquery.dataViewer"
   ```

#### Error: "Query exceeded bytes billed limit"
**Symptoms**: Queries fail with quota exceeded error

**Resolution**:
1. Check current query costs in BigQuery console
2. Increase or remove `MAX_BYTES_BILLED` limit
3. Consider optimizing queries:
   - Reduce `n_results` parameter
   - Use more specific filters
   - Enable result caching

#### High Query Costs
**Symptoms**: BigQuery costs higher than expected

**Resolution**:
1. Review query patterns in BigQuery console
2. Enable query result caching (default 24 hours)
3. Consider table partitioning for large datasets
4. Evaluate if ChromaDB would be more cost-effective for your use case

### Index Rebuild

If embeddings model changes or data quality issues occur, rebuild the vector index:

1. Drop existing index:
   ```sql
   DROP VECTOR INDEX documents_embedding_idx
   ON `cal-rag-agent.calrag.documents`;
   ```

2. Recreate index with new parameters:
   ```sql
   CREATE VECTOR INDEX documents_embedding_idx
   ON `cal-rag-agent.calrag.documents`(embedding)
   OPTIONS(
     distance_type='COSINE',
     index_type='IVF',
     ivf_options='{"num_lists": 1000}'
   );
   ```

3. Monitor index build progress:
   ```sql
   SELECT index_name, index_status, coverage_percentage
   FROM `cal-rag-agent.calrag.INFORMATION_SCHEMA.VECTOR_INDEXES`
   WHERE table_name = 'documents';
   ```

### Switching Between Backends

#### From ChromaDB to BigQuery
1. Ensure BigQuery table is populated (see ingestion scripts)
2. Update environment:
   ```bash
   export VECTOR_BACKEND=bigquery
   ```
3. Restart application
4. Verify diagnostics show BigQuery backend
5. Run smoke tests

#### From BigQuery to ChromaDB
1. Update environment:
   ```bash
   export VECTOR_BACKEND=chroma
   # or unset VECTOR_BACKEND
   ```
2. Ensure Chroma data is available at CHROMA_DIR
3. Restart application
4. Verify diagnostics show Chroma backend

### Data Ingestion (BigQuery)

**Note**: Real-time insertion via `insert_from_url()` is not supported for BigQuery backend. Use batch ingestion instead.

#### Batch Ingestion
1. Prepare data in JSONL format with required fields:
   ```json
   {"chunk_id": "...", "content": "...", "embedding": [...], "source_url": "...", "metadata": {...}}
   ```

2. Load to BigQuery:
   ```bash
   bq load --source_format=NEWLINE_DELIMITED_JSON \
     cal-rag-agent:calrag.documents \
     data.jsonl \
     schema.json
   ```

3. Verify ingestion:
   ```bash
   python scripts/verify_bigquery.py
   ```

### Emergency Rollback to ChromaDB

If BigQuery issues occur in production:

1. **Immediate**: Set environment variable:
   ```bash
   export VECTOR_BACKEND=chroma
   ```

2. **Restart** the application (Cloud Run, App Hosting, or local)

3. **Verify**: Check diagnostics sidebar shows "chroma" backend

4. **Monitor**: Ensure queries work as expected

5. **Investigate** BigQuery issue offline

### Health Checks

Add to your monitoring:

**ChromaDB Health Check**:
```bash
# Check collection count
curl http://localhost:8501/_stcore/health

# Verify in UI Diagnostics:
# - Collection name: docs_ibc_v2
# - Document count > 0
# - Chroma dir ready: true
```

**BigQuery Health Check**:
```bash
# Run verification
python scripts/verify_bigquery.py

# Expected: exit code 0
# If exit code 1: check logs for specific error
```

### Cost Estimation

#### BigQuery Costs (Example)
- **Storage**: 100GB data × $0.02/GB/month = $2/month
- **Queries**: 1000 queries/day × 1GB scanned × $5/TB = ~$0.15/day = $4.50/month
- **Index storage**: Additional ~20-50% of base storage cost
- **Total estimate**: ~$10-15/month for moderate usage

#### ChromaDB Costs (Example)
- **Storage**: Local disk space (varies by deployment)
- **Compute**: Memory/CPU overhead (included in container costs)
- **Network**: None (local queries)
- **Total**: $0 additional (beyond base infrastructure)

### Contacts & Escalation

- **BigQuery Issues**: Check Cloud Console → Support
- **Migration Questions**: See MIGRATION_NOTES.md
- **Performance Tuning**: Review scripts/test_bq_vector.py benchmark results
