# 🎯 PRECISION RAG

**Reliable Enterprise Retrieval-Augmented Generation**

A production-grade, self-hostable RAG platform with full knowledge-base isolation, multi-format ingestion, hybrid retrieval, citation verification, and a polished enterprise UI.

---

## ✨ Features

| Feature | Details |
|---|---|
| **Knowledge Bases** | Isolated document collections — each with its own dense + BM25 index |
| **Multi-format ingestion** | Markdown, TXT, HTML, PDF, DOCX, PPTX, XLSX, CSV, JSON |
| **Three chunking strategies** | Fixed-window, Recursive (paragraph-aware), Semantic (embedding-based) |
| **Hybrid retrieval** | Dense (OpenAI embeddings) + BM25, fused via Reciprocal Rank Fusion |
| **Cross-encoder reranking** | LLM-as-judge or `sentence-transformers` cross-encoder |
| **Grounded generation** | Bracketed inline citations `[1]`, `[2]` with source metadata |
| **Citation verification** | LLM judge scores each citation for factual grounding |
| **IDK gate** | Refuses to answer when retrieval confidence is too low |
| **Document dedup** | SHA-256 content hash prevents re-ingesting the same file |
| **Per-KB settings** | Chunk size, overlap, top-k, reranker, IDK threshold — all per KB |
| **Real health dashboard** | Component status derived from actual system state |
| **Enterprise UI** | Dark-mode Streamlit UI with KB management, chunk inspector, chat |

---

## 🏗 Architecture

```
User → Streamlit UI (port 8501)
              ↕ HTTP
         FastAPI API (port 8100)
              ↕
    ┌─────────────────────────┐
    │       RagEngine         │
    │  ┌───────────────────┐  │
    │  │  IngestionPipeline│  │  ← load → chunk → embed → dedup → index
    │  └───────────────────┘  │
    │  ┌───────────────────┐  │
    │  │  HybridRetriever  │  │  ← dense ANN + BM25 + RRF + reranker
    │  └───────────────────┘  │
    │  ┌───────────────────┐  │
    │  │  GroundedAnswerer │  │  ← prompt + generate + cite + verify + IDK
    │  └───────────────────┘  │
    └─────────────────────────┘
              ↕
    ┌─────────────────────────┐
    │  Storage Layer          │
    │  KnowledgeBaseStore     │  ← data/precision-rag-data/knowledge_bases.json
    │  DocumentStore          │  ← data/precision-rag-data/docs_<kb_id>.json
    │  ChunkMetaStore         │  ← data/precision-rag-data/chunks_<kb_id>.json
    │  DenseVectorStore       │  ← data/indexes/<kb_id>/recursive/dense.json
    │  BM25Store              │  ← data/indexes/<kb_id>/recursive/sparse.json
    └─────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Install

```bash
# Core + enterprise format support (DOCX, PPTX, XLSX) + UI
pip install -e ".[enterprise,ui]"

# With dev tools
pip install -e ".[enterprise,ui,dev]"
```

### 2. Configure

```bash
# Required: your OpenAI API key
export RAG_OPENAI_API_KEY=sk-...

# Optional overrides (see .env.example for full list)
export RAG_GENERATION_MODEL=gpt-4o
export RAG_EMBEDDING_MODEL=text-embedding-3-small
export RAG_RERANK_KIND=llm           # llm | cross-encoder | none
export RAG_IDK_RETRIEVAL_THRESHOLD=0.35
```

Copy `.env.example` to `.env` and edit as needed — it's auto-loaded.

### 3. Start the backend

```bash
rag serve
# or
python scripts/start_server.py
```

API: http://localhost:8100  
Swagger docs: http://localhost:8100/docs

### 4. Start the UI

```bash
streamlit run src/rag/ui.py
```

UI: http://localhost:8501

### 5. Load demo data

With the backend running:

```bash
python scripts/load_demo_data.py
```

This creates an **"Acme HR Policies"** knowledge base with 5 enterprise documents (employee handbook, leave policy, IT security policy, expense policy, travel policy).

---

## 📖 Usage

### CLI

```bash
# Ingest documents (into the default global index)
rag ingest --path ./docs/

# Ask a question (against the default index)
rag ask --question "What is the maternity leave policy?"

# Run evaluations
rag eval --cases golden_qa.jsonl

# Start the API server
rag serve

# Print resolved configuration
rag config
```

### API

```bash
# Health check
curl http://localhost:8100/health

# System stats
curl http://localhost:8100/system/stats

# Create a knowledge base
curl -X POST http://localhost:8100/knowledge-bases \
  -H "Content-Type: application/json" \
  -d '{"name": "HR Policies", "description": "Human resources documentation"}'

# Upload a document
curl -X POST http://localhost:8100/knowledge-bases/<kb_id>/documents \
  -F "file=@./handbook.pdf"

# List documents in a KB
curl http://localhost:8100/knowledge-bases/<kb_id>/documents

# Inspect chunks
curl "http://localhost:8100/knowledge-bases/<kb_id>/chunks?limit=50&search=password"

# Ask a question (global index)
curl -X POST http://localhost:8100/v1/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the sick leave entitlement?"}'
```

---

## 🗂 Project Structure

```
rag-hybrid-search/
├── src/rag/
│   ├── api/app.py              # FastAPI app (KB + legacy v1 endpoints)
│   ├── cli.py                  # Click CLI (ingest, ask, eval, serve, config)
│   ├── ui.py                   # Streamlit enterprise UI
│   ├── config.py               # Pydantic Settings (env vars / .env)
│   ├── models.py               # Core Pydantic models (Document, Chunk, Answer…)
│   ├── engine.py               # RagEngine: retrieve → generate → verify loop
│   ├── llm_client.py           # Async OpenAI-compatible HTTP client
│   ├── logging.py              # structlog JSON logging
│   ├── ingestion/
│   │   ├── pipeline.py         # IngestionPipeline: load → chunk → embed → index
│   │   ├── loaders.py          # md / txt / html / pdf loaders
│   │   └── chunkers.py         # Fixed / Recursive / Semantic chunkers
│   ├── retrieval/
│   │   ├── retriever.py        # HybridRetriever: dense + BM25 + RRF
│   │   ├── reranker.py         # LLM reranker + cross-encoder + NoOp
│   │   └── fusion.py           # Reciprocal Rank Fusion
│   ├── generation/
│   │   ├── answerer.py         # GroundedAnswerer + citation parser
│   │   └── verifier.py         # Citation verification + IDK gate
│   ├── store/
│   │   ├── dense.py            # DenseVectorStore (flat JSON + cosine ANN)
│   │   └── sparse.py           # BM25Store (rank-bm25)
│   ├── storage/                # NEW: enterprise metadata layer
│   │   ├── knowledge_base.py   # KnowledgeBaseStore (CRUD, JSON-backed)
│   │   ├── document_store.py   # DocumentStore (per-KB, status tracking)
│   │   ├── chunk_meta.py       # ChunkMetaStore (chunk inspection, search)
│   │   └── models.py           # Storage Pydantic models
│   └── services/               # NEW: business logic services
│       ├── kb_service.py       # KnowledgeBaseService (index isolation)
│       ├── ingestion_service.py # IngestionService (extended formats + metadata)
│       └── health_service.py   # HealthService (real component status)
├── tests/
│   ├── test_api.py             # Legacy API endpoint tests
│   ├── test_kb_service.py      # NEW: KB service + new API endpoint tests
│   ├── test_storage.py         # NEW: Storage layer unit tests
│   ├── test_chunkers.py
│   ├── test_loaders.py
│   ├── test_retrieval_and_generation.py
│   ├── test_stores_and_fusion.py
│   ├── test_models_and_config.py
│   └── test_llm_and_logging.py
├── demo_docs/                  # NEW: 5 enterprise sample documents
│   ├── employee_handbook.md
│   ├── leave_policy.md
│   ├── it_security_policy.md
│   ├── expense_policy.md
│   └── travel_policy.md
├── scripts/
│   ├── load_demo_data.py       # NEW: Creates demo KB + ingests docs
│   └── start_server.py        # NEW: Standalone server startup
└── pyproject.toml
```

---

## ⚙️ Configuration Reference

All settings are read from environment variables (or a `.env` file). Prefix: `RAG_`.

| Variable | Default | Description |
|---|---|---|
| `RAG_OPENAI_API_KEY` | *(required)* | OpenAI (or compatible) API key |
| `RAG_OPENAI_BASE_URL` | `https://api.openai.com/v1` | Base URL for API calls |
| `RAG_GENERATION_MODEL` | `gpt-4o` | LLM for answer generation |
| `RAG_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `RAG_JUDGE_MODEL` | `gpt-4o-mini` | LLM for citation judging |
| `RAG_RERANK_KIND` | `llm` | `llm` \| `cross-encoder` \| `none` |
| `RAG_RERANK_MODEL` | `gpt-4o-mini` | Model for LLM reranker |
| `RAG_CHUNKING_STRATEGY` | `recursive` | `recursive` \| `fixed` \| `semantic` |
| `RAG_CHUNK_SIZE_TOKENS` | `512` | Target tokens per chunk |
| `RAG_CHUNK_OVERLAP_TOKENS` | `64` | Overlap between adjacent chunks |
| `RAG_DENSE_TOP_K` | `20` | Candidates from dense retrieval |
| `RAG_SPARSE_TOP_K` | `20` | Candidates from BM25 |
| `RAG_RRF_K` | `60` | RRF rank-discount factor |
| `RAG_FINAL_TOP_K` | `5` | Passages sent to LLM after reranking |
| `RAG_IDK_RETRIEVAL_THRESHOLD` | `0.35` | Below this → "I don't know" |
| `RAG_JUDGE_WEIGHT` | `0.5` | `w·citation_acc + (1-w)·retrieval` |
| `RAG_DEDUP_COSINE_THRESHOLD` | `0.95` | Cosine similarity above → dedup |
| `RAG_INDEX_DIR` | `data/index` | Where to store index files |
| `RAG_API_HOST` | `0.0.0.0` | API server bind address |
| `RAG_API_PORT` | `8100` | API server port |
| `RAG_LOG_LEVEL` | `INFO` | Log level |

---

## 🧪 Testing

```bash
# Run all 90 tests
pytest

# With coverage
pytest --cov=rag --cov-report=term-missing

# Specific test modules
pytest tests/test_storage.py -v
pytest tests/test_kb_service.py -v
```

All tests use mock LLM clients — no real API calls are made during tests.

---

## 📐 Design Decisions

### Why flat-file JSON storage?
The current `DenseVectorStore` and `BM25Store` use flat JSON files. The new `storage/` layer mirrors this approach for maximum simplicity and zero infrastructure dependencies. Swapping to SQLite or PostgreSQL only requires implementing a new store that satisfies the same interface.

### Why not replace the retrieval engine?
The existing dense + BM25 + RRF + reranker + grounded generation pipeline was already production-quality. PRECISION RAG adds the enterprise wrapper (KB isolation, document management, metadata, UI) **around** it — not replacing it.

### Why per-KB index isolation?
Each knowledge base gets its own `DenseVectorStore` and `BM25Store` under `data/indexes/<kb_id>/`. This means queries against KB-A can never return documents from KB-B, regardless of score similarity.

### Confidence score formula
`composite = judge_weight × citation_accuracy + (1 - judge_weight) × retrieval_confidence`

The IDK gate fires when `retrieval_confidence < idk_threshold`, before LLM generation is attempted, saving tokens.

---

## 🔌 API Reference

Full Swagger UI at: http://localhost:8100/docs

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Detailed component health |
| `GET` | `/system/stats` | System-wide KB/doc/chunk counts |
| `GET` | `/knowledge-bases` | List all knowledge bases |
| `POST` | `/knowledge-bases` | Create a knowledge base |
| `GET` | `/knowledge-bases/{id}` | Get KB details |
| `PATCH` | `/knowledge-bases/{id}` | Update KB name/description |
| `PUT` | `/knowledge-bases/{id}/settings` | Update KB retrieval settings |
| `DELETE` | `/knowledge-bases/{id}` | Delete KB + all its data |
| `GET` | `/knowledge-bases/{id}/documents` | List documents |
| `POST` | `/knowledge-bases/{id}/documents` | Upload + ingest document |
| `GET` | `/knowledge-bases/{id}/documents/{doc_id}` | Get document |
| `DELETE` | `/knowledge-bases/{id}/documents/{doc_id}` | Delete document |
| `POST` | `/knowledge-bases/{id}/documents/{doc_id}/reindex` | Re-ingest document |
| `GET` | `/knowledge-bases/{id}/chunks` | List chunks (filterable) |
| `POST` | `/knowledge-bases/{id}/ingest` | Ingest server-side file path |
| `POST` | `/v1/ask` | Ask question (global index) |
| `POST` | `/v1/ingest` | Ingest path (global index) |

---

## 🗺 Roadmap (Phase 2)

- [ ] Cross-KB federated search
- [ ] Evaluation dashboard with golden Q&A sets
- [ ] SQLite-backed storage for production deployments
- [ ] Streaming `/v1/ask` response (SSE)
- [ ] KB-scoped chat (query specific KB, not global index)
- [ ] User authentication + multi-tenancy
- [ ] Async background ingestion with progress polling
- [ ] Auto-chunking strategy selection based on document type

---

## License

MIT
