# BigQuery Vector Search Migration Notes

## Date: 2025-11-19

## Migration Summary

Successfully migrated from ChromaDB-only vector storage to a dual-backend architecture supporting both ChromaDB and BigQuery Vector Search.

## What Changed

### Architecture
- **Added**: Vector store abstraction layer (`utils_vectorstore.py`)
- **Added**: BigQuery helper module (`utils_bigquery.py`)
- **Modified**: `rag_agent.py` to use vector store abstraction
- **Modified**: `streamlit_app.py` to support both backends
- **Added**: `VECTOR_BACKEND` environment variable to toggle backends

### New Environment Variables

```bash
# Vector storage backend selection
VECTOR_BACKEND=chroma  # or "bigquery"

# BigQuery configuration
BQ_PROJECT=cal-rag-agent
BQ_DATASET=calrag
BQ_TABLE=documents
```

### Files Created
1. `utils_bigquery.py` - BigQuery operations (vector_search, keyword_search, etc.)
2. `utils_vectorstore.py` - Abstraction layer with VectorStore interface
3. `scripts/verify_bigquery.py` - Verification script for BigQuery setup
4. `scripts/test_bq_vector.py` - Test suite for BigQuery vector operations
5. `PHASE0_CURRENT_STATE.md` - Documentation of pre-migration state
6. `MIGRATION_NOTES.md` - This file

### Files Modified
1. `requirements.txt` - Added `google-cloud-bigquery>=3.10.0`
2. `rag_agent.py` - Refactored to use vector store abstraction
3. `streamlit_app.py` - Updated for multi-backend support
4. `.env.example` - Added BigQuery configuration

## How to Use

### Using ChromaDB (Default)
No changes needed. The system defaults to ChromaDB:

```bash
# Option 1: Leave VECTOR_BACKEND unset (defaults to chroma)
# Option 2: Explicitly set
export VECTOR_BACKEND=chroma
```

### Using BigQuery Vector Search
Set the environment variable and ensure BigQuery credentials are configured:

```bash
export VECTOR_BACKEND=bigquery
export BQ_PROJECT=cal-rag-agent
export BQ_DATASET=calrag
export BQ_TABLE=documents

# Ensure Google Cloud credentials are available
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json

# Run the application
streamlit run streamlit_app.py
```

## BigQuery Setup Requirements

### 1. Table Schema
The BigQuery table must have:
- `chunk_id` (STRING) - Unique identifier
- `content` (STRING) - Document text
- `embedding` (ARRAY<FLOAT64>) - Vector embeddings
- `source_url` (STRING) - Source URL or file path
- `metadata` (JSON or STRING) - Metadata as JSON

### 2. Vector Index (Recommended)
For optimal performance, create a vector index:

```sql
CREATE VECTOR INDEX documents_embedding_idx
ON `cal-rag-agent.calrag.documents`(embedding)
OPTIONS(distance_type='COSINE', index_type='IVF');
```

### 3. Permissions
The service account needs:
- `bigquery.tables.get`
- `bigquery.tables.getData`
- `bigquery.jobs.create`

Roles: `roles/bigquery.dataViewer` + `roles/bigquery.jobUser`

## Testing

### Verify BigQuery Setup
```bash
python scripts/verify_bigquery.py
```

### Run BigQuery Vector Tests
```bash
python scripts/test_bq_vector.py
```

## Rollback Plan

If issues occur with BigQuery:

```bash
# Revert to Chroma immediately
export VECTOR_BACKEND=chroma

# Or remove the environment variable entirely
unset VECTOR_BACKEND
```

## Known Limitations

### BigQuery Backend
1. **No real-time insertion**: Use batch ingestion scripts instead of `insert_from_url()`/`insert_from_file()`
2. **Query costs**: Each vector search incurs BigQuery query costs
3. **Latency**: Slightly higher latency than local Chroma (network overhead)

### Chroma Backend
1. **Scalability**: Limited by local disk and memory
2. **Multi-instance**: Requires shared filesystem for multi-replica deployments

## Performance Considerations

### BigQuery
- **Pros**: Scales to billions of vectors, managed service, no local storage
- **Cons**: Query costs, network latency, requires internet connectivity
- **Best for**: Production deployments, large datasets (>100M chunks)

### ChromaDB
- **Pros**: Fast local queries, no API costs, works offline
- **Cons**: Resource intensive, single-node limitation
- **Best for**: Development, testing, small datasets (<10M chunks)

## Monitoring

### BigQuery Queries
Monitor via Cloud Console:
- Go to BigQuery > Query History
- Check bytes processed and execution time
- Set up cost alerts if needed

### Application Logs
Both backends emit structured JSON logs:

```json
{
  "where": "bigquery" or "chroma",
  "action": "vector_search",
  "results_count": 5,
  "elapsed_ms": 234
}
```

## Cost Management

### BigQuery Vector Search Costs
- **Queries**: ~$5-10 per TB processed
- **Storage**: ~$20 per TB per month
- **Vector index**: Additional storage costs

### Optimization Tips
1. Use `MAX_BYTES_BILLED` to cap query costs
2. Enable query result caching
3. Monitor INFORMATION_SCHEMA.JOBS for expensive queries
4. Consider partitioning large tables by date

## Future Enhancements

Potential improvements for future iterations:
1. Add Pinecone, Weaviate, or other vector DB backends
2. Implement hybrid search (BM25 + vector)
3. Add automatic index optimization
4. Support multi-backend federation
5. Add benchmark suite for backend comparison

## Troubleshooting

### "VECTOR_BACKEND=bigquery but no data found"
- Verify table exists: `scripts/verify_bigquery.py`
- Check service account permissions
- Ensure data was ingested to BigQuery

### "ModuleNotFoundError: No module named 'google'"
- Install dependencies: `pip install google-cloud-bigquery>=3.10.0`
- Or run: `pip install -r requirements.txt`

### "ChromaDB working but BigQuery fails"
- Check credentials: `gcloud auth application-default login`
- Verify project ID matches BQ_PROJECT
- Check firewall/VPC rules if in restricted environment

## Contact

For issues or questions about the migration:
- Check GitHub Issues
- Review OPERATIONS_RUNBOOK.md
- Consult PHASE0_CURRENT_STATE.md for pre-migration baseline
