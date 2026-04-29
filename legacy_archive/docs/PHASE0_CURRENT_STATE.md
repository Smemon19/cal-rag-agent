# Phase 0: Current State Documentation

## Date: 2025-11-19

### BigQuery Table Information

**Target Table**: `cal-rag-agent.calrag.documents`
- **Expected Index**: `documents_embedding_idx`
- **Note**: Verification script created at `scripts/verify_bigquery.py` (to be run after dependencies installed)

### Current Environment Variables

The following environment variables are currently used by the application:

#### Core Configuration
- `OPENAI_API_KEY` - OpenAI API key for embeddings and LLM
- `MODEL_CHOICE` - LLM model choice (default: gpt-4.1-mini)

#### Chroma-Specific (TO BE DEPRECATED)
- `RAG_COLLECTION_NAME` - ChromaDB collection name (default: docs_ibc_v2)
- `CHROMA_DIR` - Directory for ChromaDB persistence
- `KEYWORD_SCAN_LIMIT` - Limit for keyword search scans (default: 2000)
- `KEYWORD_PREFILTER_LIMIT` - Prefilter limit (default: 5000)

#### Embedding Configuration (KEEP)
- `EMBEDDING_BACKEND` - Backend for embeddings: "sentence" or "openai" (default: sentence)
- `SENTENCE_MODEL` - Sentence transformers model (default: all-MiniLM-L6-v2)
- `OPENAI_EMBED_MODEL` - OpenAI embeddings model (default: text-embedding-3-large)
- `EMBED_CACHE_SIZE` - Embedding cache size (default: 2048)
- `EMBED_BATCH_SIZE` - Batch size for embeddings (default: 64)

#### Hugging Face/Transformers Caching (KEEP)
- `HF_HOME` - Hugging Face cache directory
- `TRANSFORMERS_CACHE` - Transformers cache directory
- `SENTENCE_TRANSFORMERS_HOME` - Sentence transformers cache directory
- `TORCH_HOME` - PyTorch cache directory

#### New BigQuery Variables (TO BE ADDED)
- `VECTOR_BACKEND` - Vector storage backend: "chroma" or "bigquery" (new)
- `BQ_PROJECT` - BigQuery project ID (default: cal-rag-agent)
- `BQ_DATASET` - BigQuery dataset ID (default: calrag)
- `BQ_TABLE` - BigQuery table name (default: documents)

### Current Architecture

#### Vector Storage
- **Current**: ChromaDB with local persistence
- **Location**: `chroma_db/` directory or `~/.calrag/chroma`
- **Collection**: Managed via `RAG_COLLECTION_NAME` environment variable

#### Key Files
1. **utils.py** (1104 lines)
   - Contains all Chroma-related operations
   - Functions to refactor:
     - `get_chroma_client()` - Client initialization
     - `get_or_create_collection()` - Collection management
     - `add_documents_to_collection()` - Document insertion
     - `query_collection()` - Vector search
     - `keyword_search_collection()` - Keyword fallback
     - `get_existing_ids()` - Duplicate detection
   - Caches:
     - `_CHROMA_CLIENT_CACHE` - Client instances
     - `_COLLECTION_CACHE` - Collection instances
     - `_LAST_VECTOR_STORE_STATUS` - Status tracking

2. **rag_agent.py** (~600 lines estimated)
   - Main RAG agent class `RagAgent`
   - Dependencies: `RAGDeps` dataclass with chroma_client
   - Methods to update:
     - `__init__()` - Constructor
     - `insert_from_url()` - URL ingestion
     - `insert_from_file()` - File ingestion
     - `retrieve()` - Retrieval logic

3. **streamlit_app.py**
   - Streamlit UI for querying
   - Uses Chroma diagnostics
   - Needs BigQuery diagnostics integration

4. **insert_docs.py**
   - Document ingestion CLI
   - Currently writes to Chroma
   - May need BigQuery ingestion path

### Metadata Schema (Standardized)

Current chunks include:
- `source_url` - Canonical URL or file:// path
- `source_type` - One of: web, pdf, markdown, txt, other
- `section_path` - Hierarchy like "H1 > H2 > H3"
- `headers` - Raw header text
- `page_number` - PDF page number (if applicable)
- `char_count` / `word_count` - Content statistics
- `chunk_id` - Deterministic ID from source + section + page + index
- `inserted_at` - UTC ISO timestamp
- `content_preview` - First ~180 characters
- `embedding_backend` / `embedding_model` - Embedding configuration
- `mime_type` - Content type
- `title` - Document title

### Current Streamlit UI Behavior

The Streamlit app (streamlit_app.py) provides:
- Chat interface for querying documentation
- Sidebar filters:
  - Header contains (section filtering)
  - Source contains (source filtering)
- Diagnostics:
  - Collection information
  - Embedding backend/model
  - Memory usage
  - Query count
  - Chroma client status

### Dependencies

#### Current (requirements.txt)
- streamlit==1.37.1
- chromadb==0.5.3 (TO BE MADE OPTIONAL/REMOVED)
- sentence-transformers==3.0.1
- transformers==4.44.2
- huggingface-hub==0.24.6
- torch>=2.2,<3.0
- python-dotenv==1.0.1
- openai==1.43.0
- pydantic==2.8.2
- tenacity==8.3.0
- pydantic-ai
- psutil
- more-itertools
- pandas
- PyMuPDF
- pytesseract
- Pillow
- crawl4ai
- requests
- beautifulsoup4
- pytest

#### Added
- google-cloud-bigquery>=3.10.0 (ADDED)

### Dockerfile

Current Dockerfile:
- Base: python:3.11-slim
- Includes tesseract-ocr for PDF OCR
- Sets HF_HOME, SENTENCE_TRANSFORMERS_HOME to /app/models
- Exposes port 8080 for Cloud Run
- Runs streamlit on 0.0.0.0:${PORT}
- **Needs**: BigQuery client library addition (already in requirements.txt)

### Identified Tasks

1. ✅ Document current environment variables
2. ✅ Document current architecture
3. ⏳ Create utils_bigquery.py module
4. ⏳ Create vector store abstraction layer
5. ⏳ Refactor Chroma code into ChromaVectorStore class
6. ⏳ Implement BigQueryVectorStore class
7. ⏳ Update rag_agent.py
8. ⏳ Update streamlit_app.py
9. ⏳ Update environment configuration
10. ⏳ Test locally and in Cloud Run

### Migration Strategy Notes

- **Backwards Compatibility**: Keep Chroma support via VECTOR_BACKEND toggle
- **Gradual Migration**: Allow switching between backends without code changes
- **Testing**: Verify BigQuery works before removing Chroma artifacts
- **Rollback**: Can revert to Chroma by setting VECTOR_BACKEND=chroma

## Next Steps

Proceed to Phase 1: Create BigQuery helper module (utils_bigquery.py)
