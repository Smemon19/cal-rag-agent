# Pydantic AI Documentation Crawler & RAG Agent

An intelligent documentation crawler and retrieval-augmented generation (RAG) system, powered by Crawl4AI and Pydantic AI. This project enables you to crawl, chunk, and vectorize documentation from any website, `.txt`/Markdown pages (llms.txt), or sitemap, and interact with the knowledge base using a Streamlit interface.

---

## Features

- **Flexible documentation crawling:** Handles regular websites, `.txt`/Markdown pages (llms.txt), and sitemaps.
- **Parallel and recursive crawling:** Efficiently gathers large doc sites with memory-adaptive batching.
- **Smart chunking:** Hierarchical Markdown chunking by headers, ensuring chunks are optimal for vector search.
- **Vector database integration:** Stores chunks and metadata in ChromaDB for fast semantic retrieval.
- **Streamlit RAG interface:** Query your documentation with LLM-powered semantic search.
- **Extensible examples:** Modular scripts for various crawling and RAG workflows.

---

## Prerequisites

- Python 3.11+
- OpenAI API key (for embeddings and LLM-powered search)
- Crawl4AI/Playwright and other dependencies in `requirements.txt`
- (Optional) Streamlit for the web interface

---

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/coleam00/ottomator-agents.git
   cd ottomator-agents/crawl4AI-agent-v2
   ```

2. **Install dependencies:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   playwright install
   ```

3. **Set up environment variables:**
   - Copy `.env.example` to `.env`
   - Edit `.env` with your API keys and preferences:
     ```env
     OPENAI_API_KEY=your_openai_api_key
     MODEL_CHOICE=gpt-4.1-mini  # or your preferred OpenAI model
     ```

---

## Usage

### 1. Crawling and Inserting Documentation

The main entry point for crawling and vectorizing documentation is [`insert_docs.py`](insert_docs.py):

#### Supported URL Types

- **Regular documentation sites:** Recursively crawls all internal links, deduplicates by URL (ignoring fragments).
- **Markdown or .txt pages (such as llms.txt):** Fetches and chunks Markdown content.
- **Sitemaps (`sitemap.xml`):** Batch-crawls all URLs listed in the sitemap.

#### Example Usage

```bash
python insert_docs.py <URL> [--collection mydocs] [--db-dir ./chroma_db] [--embedding-model all-MiniLM-L6-v2] [--chunk-size 1000] [--max-depth 3] [--max-concurrent 10] [--batch-size 100]
```

**Arguments:**

- `URL`: The root URL, .txt file, or sitemap to crawl.
- `--collection`: ChromaDB collection name. Overrides environment for a single run.
- `--db-dir`: Directory for ChromaDB data (default: `./chroma_db`)
- `--embedding-model`: Embedding model for vector storage (default: `all-MiniLM-L6-v2`)
- `--chunk-size`: Maximum characters per chunk (default: `1000`)
- `--max-depth`: Recursion depth for regular URLs (default: `3`)
- `--max-concurrent`: Max parallel browser sessions (default: `10`)
- `--batch-size`: Batch size for ChromaDB insertion (default: `100`)

**Examples for each type (regular URL, .txt, sitemap):**

```bash
python insert_docs.py https://ai.pydantic.dev/
python insert_docs.py https://ai.pydantic.dev/llms-full.txt
python insert_docs.py https://ai.pydantic.dev/sitemap.xml
```

#### Chunking Strategy

- Splits content first by `#`, then by `##`, then by `###` headers.
- If a chunk is still too large, splits by character count.
- All chunks are less than the specified `--chunk-size` (default: 1000 characters).

#### Metadata

Each chunk is stored with:

- Source URL
- Chunk index
- Extracted headers
- Character and word counts

---

### 2. Example Scripts

The `crawl4AI-examples/` folder contains modular scripts illustrating different crawling and chunking strategies:

- **`3-crawl_sitemap_in_parallel.py`:** Batch-crawls a list of URLs from a sitemap in parallel with memory tracking.
- **`4-crawl_llms_txt.py`:** Crawls a Markdown or `.txt` file, splits by headers, and prints chunks.
- **`5-crawl_site_recursively.py`:** Recursively crawls all internal links from a root URL, deduplicating by URL (ignoring fragments).

You can use these scripts directly for experimentation or as templates for custom crawlers.

---

### 3. Running the Streamlit RAG Interface

After crawling and inserting docs, launch the Streamlit app for semantic search and question answering:

```bash
streamlit run streamlit_app.py
```

- The interface will be available at [http://localhost:8501](http://localhost:8501)
- Query your documentation using natural language and get context-rich answers.

---

## Project Structure

```
crawl4AI-agent-v2/
├── crawl4AI-examples/
│   ├── 3-crawl_docs_FAST.py
│   ├── 4-crawl_and_chunk_markdown.py
│   └── 5-crawl_recursive_internal_links.py
├── insert_docs.py
├── rag_agent.py
├── streamlit_app.py
├── utils.py
├── requirements.txt
├── .env.example
└── README.md
```

---

## Ingestion & Retrieval Flow

The system ingests URLs (regular, sitemap, or Markdown/txt) and local files (PDF or text). Type detection routes inputs to Crawl4AI-based crawling or to a PDF pipeline that extracts text, images, and OCR, then merges and chunks content. All sources are smart-chunked and enriched with standardized metadata before being inserted in batches into a single ChromaDB Collection created/opened with the selected embedding backend/model. The Streamlit chat UI calls a Pydantic AI agent whose typed retrieve tool performs hybrid retrieval (vector first, keyword fallback for exact literals/sections), merges results, applies optional filters, formats context, and streams the final LLM answer back to the UI.

```mermaid
graph LR
  %% ===== Ingestion =====
  subgraph Ingestion
    I[Input: URL / sitemap / md|txt URL / local file (PDF or text)]
    D[Type detection]
    I --> D

    %% Web ingestion
    D -->|sitemap URL| S1[Parse sitemap]
    S1 --> S2[Crawl batch (Crawl4AI + MemoryAdaptiveDispatcher)]
    D -->|md/txt URL| M1[Fetch rendered content (Crawl4AI)]
    D -->|regular URL| W1[Crawl & render to Markdown (Crawl4AI)]

    %% PDF ingestion
    D -->|local PDF| P1[Extract visible text per page]
    P1 --> P2[Extract images per page]
    P2 --> P3[OCR images]
    P3 --> P4[Merge text + OCR per page]
    P4 --> P5[Chunk merged text]

    %% Local text files
    D -->|local text file| F1[Read file as markdown]

    %% Chunking & metadata
    W1 --> C1[Smart chunk markdown]
    M1 --> C1
    F1 --> C1
    P5 --> C2[Standardize metadata]
    C1 --> C2
    C2 --> C3[Insert chunks (batched)]
    C3 --> C4[Create/Open collection (selected embedding backend/model)]
  end

  %% Central storage
  C4 --> COL[ChromaDB Collection]

  %% ===== Retrieval =====
  subgraph Retrieval
    UI[Streamlit Chat UI] --> AG[Pydantic AI Agent (typed retrieve tool)]
    AG -->|vector query| COL
    AG -->|keyword fallback| COL
    COL --> MR[Merge results]
    MR --> FLT[Optional filters: header_contains, source_contains]
    FLT --> CTX[Format results as context]
    CTX --> LLM[LLM answer (streamed tokens)]
    LLM --> UI
  end
```

---

## Configuration: Collection Name

- **RAG_COLLECTION_NAME**: Controls the default ChromaDB collection for both ingestion and the Streamlit UI.
  - If unset or blank (whitespace only), the system falls back to `docs`.
  - The ingestion CLI supports `--collection <name>` to override the environment for a single run.

Examples:

```bash
# Use default (docs)
unset RAG_COLLECTION_NAME
python insert_docs.py https://example.com

# Set env for both ingest and UI
export RAG_COLLECTION_NAME=building-code
python insert_docs.py https://example.com
streamlit run streamlit_app.py

# Override just for this ingest run
export RAG_COLLECTION_NAME=building-code
python insert_docs.py https://example.com --collection my-temp-collection
```

On startup, both ingestion and the UI log the resolved collection name once.

---

## Optional Retrieval Filters

You can refine results using two optional filters that apply after retrieval and before the LLM sees the context:

- `header_contains` (case-insensitive): Match a substring in the section header/path. Fallback order in metadata: `section_path` → `headers` → `title` → `header`.
- `source_contains` (case-insensitive): Match a substring in the source identifier. Fallback order in metadata: `source_url` → `source` → `file_path`.

Notes:

- Empty/whitespace values are ignored. Very short filters (e.g., 1 character) can be noisy.
- Filtering happens after vector+keyword merging and before final truncation.
- If no results match, the context includes a brief notice suggesting you remove or relax filters.

### Streamlit UI

- Use the sidebar fields “Header contains” and “Source contains”. Leave blank for default behavior.

### Programmatic usage

If calling the retrieve tool directly, pass parameters:

```python
await retrieve(context, "your question", n_results=5, header_contains="Roofing", source_contains="pydantic.dev")
```

Filtering works best when ingestion stores `section_path` and `source_url` metadata, but it degrades gracefully if missing.

---

## Embedding Backends

Choose the embedding provider via environment variables (centralized; no code changes needed in call sites):

- `EMBEDDING_BACKEND`: `sentence` (default) or `openai`. Invalid values fall back to `sentence` with a warning.
- `SENTENCE_MODEL`: sentence-transformers model (default: `all-MiniLM-L6-v2`).
- `OPENAI_EMBED_MODEL`: OpenAI embeddings model (default: `text-embedding-3-large`). Requires `OPENAI_API_KEY`.

Examples:

```bash
# Default: local sentence-transformers
unset EMBEDDING_BACKEND SENTENCE_MODEL OPENAI_EMBED_MODEL
streamlit run streamlit_app.py

# Use a different sentence-transformers model
export EMBEDDING_BACKEND=sentence
export SENTENCE_MODEL=all-MiniLM-L12-v2
streamlit run streamlit_app.py

# Switch to OpenAI embeddings
export EMBEDDING_BACKEND=openai
export OPENAI_API_KEY=sk-...
# optional override
export OPENAI_EMBED_MODEL=text-embedding-3-small
streamlit run streamlit_app.py
```

Notes and tradeoffs:

- Local (sentence-transformers) costs nothing and works offline; larger models may use significant memory.
- OpenAI embeddings can improve quality in some domains; requires API key and network access.
- You can switch backends without reingesting for testing, but mixing embeddings inside the same Chroma collection is not recommended and may degrade retrieval. Prefer separate collections per backend.

On startup, both ingestion and the UI log which backend and model are active.

---

## Metadata Schema for Chunks

New ingests populate a standardized metadata schema to improve filtering and provenance. Legacy chunks still work via fallbacks, but reingestion is recommended for best results.

- **source_url**: Canonical URL (normalized) or `file://` absolute path.
- **source_type**: One of `web`, `pdf`, `markdown`, `txt`, `other`.
- **section_path**: Human-readable hierarchy like `H1 > H2 > H3` (empty if not available).
- **headers**: Raw header text found in the chunk.
- **page_number**: PDF 1-based page number for the chunk; empty otherwise.
- **char_count / word_count**: Character and word counts of the chunk content.
- **chunk_id**: Stable deterministic id from source, section, page, and local index.
- **inserted_at**: UTC ISO timestamp at ingestion time.
- **content_preview**: First ~180 characters for quick UI previews.
- **embedding_backend / embedding_model**: Copied from embeddings selection for audits.
- **mime_type**: e.g., `text/markdown`, `application/pdf`, `text/plain`.
- **title**: Best available title (web H1/HTML title, PDF metadata or filename, Markdown H1).

Filtering leverages these fields via fallbacks (headers: `section_path` → `headers` → `title` → `header`; source: `source_url` → `source` → `file_path`).

Example benefit:

```
section_path: Roofing > Section 1507 > Asphalt Shingles
source_url: https://example.com/building-code/roofing
```

These make `header_contains=Shingles` and `source_contains=roofing` precise and robust.

---

## Advanced Usage & Customization

- **Chunking:** Tune `--chunk-size` for your retrieval use case.
- **Embeddings:** Swap out the embedding model with `--embedding-model`.
- **Crawling:** Adjust `--max-depth` and `--max-concurrent` for large sites.
- **Vector DB:** Use your own ChromaDB directory or collection for multiple projects.

---

## Troubleshooting

- Ensure all dependencies are installed and environment variables are set.
- For large sites, increase memory or decrease `--max-concurrent`.
- If you encounter crawling issues, try running the example scripts for isolated debugging.

---

## Evaluation

A lightweight evaluation suite lives in `eval/`.

- Run ad-hoc evaluation and write a report:

```bash
make eval
```

- Run smoke tests (deterministic string checks for critical rules):

```bash
make tests
```

The eval CLI (`eval/runner.py`) reads `eval/dataset.jsonl`, calls the agent for each
question, scores using `eval/rubric.py`, prints per-item scores, and writes `eval/report.md`.
