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
