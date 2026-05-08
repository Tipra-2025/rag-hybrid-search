# Changelog

## [0.1.0] — Initial release

### Added
- Strict Pydantic v2 contracts (`Document`, `Chunk`, `DenseHit`, `SparseHit`, `FusedHit`, `RankedHit`, `Citation`, `Answer`, `EvalCase`).
- Multi-format ingestion loaders: markdown, txt, html (BeautifulSoup), pdf (pypdf).
- Three swappable chunking strategies behind one `Chunker` ABC: fixed-token, recursive-character, semantic (with embedding callback + sync fallback).
- `IngestionPipeline`: chunk → cosine-dedup (≥ 0.95) → dual-index. Incremental and idempotent (content-hash skip).
- Dense store: JSON-backed, NumPy cosine top-K, near-duplicate gate.
- Sparse store: `rank_bm25.BM25Okapi` rebuilt on corpus change, JSON persistence.
- Reciprocal Rank Fusion (`k = 60` default) over the two retrievers.
- Pluggable reranker: `NoOpReranker`, `LLMReranker` (default), `CrossEncoderReranker` (optional `[reranker]` extra).
- Grounded answerer with bracketed citations and hard IDK gate.
- Citation parser + LLM-as-judge verifier returning per-citation `supported` booleans.
- Composite confidence blending retrieval and citation accuracy.
- FastAPI gateway (`POST /v1/ask`, `POST /v1/ingest`, `GET /health`).
- Click CLI: `rag ingest`, `rag ask`, `rag eval`, `rag serve`, `rag config`.
- Streamlit UI (optional `[ui]` extra) with side-by-side citations and post-rerank chunks.
- 50+-test pytest suite (`httpx.MockTransport` + deterministic pseudo-embeddings, no real network).
- mypy strict, ruff lint, GitHub Actions matrix py3.11 / py3.12.
- Dockerfile + docker-compose with optional UI profile, three architecture decision records.
- 5-row sample golden Q&A eval set under `eval/golden_qa.jsonl`.
