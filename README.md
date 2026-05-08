# rag-hybrid-search

> A production-grade RAG pipeline. Multi-format ingestion → three swappable chunking strategies → dual indexing (dense + BM25) → Reciprocal Rank Fusion → cross-encoder rerank → grounded generation with bracketed citations → LLM-as-judge citation verification → composite-confidence "I don't know" gate.

[![Tests](https://github.com/metehanulusoy/rag-hybrid-search/actions/workflows/test.yml/badge.svg)](https://github.com/metehanulusoy/rag-hybrid-search/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Type checked: mypy](https://img.shields.io/badge/typed-mypy%20strict-blueviolet.svg)](https://mypy-lang.org/)

`rag-hybrid-search` (CLI: `rag`) is a small, opinionated Python package that turns a directory of mixed-format documents into a queryable, citation-verified RAG service. It is designed for teams that want the *production* boxes ticked — strict Pydantic contracts, hybrid retrieval that doesn't depend on per-query score calibration, evidence-backed citations, and a service that refuses to answer rather than hallucinate when the corpus doesn't support it.

---

## Why this exists

Most "vector search + LLM" tutorials skip the parts that matter at scale:

- A single chunking strategy that breaks on the document type you actually have.
- Pure dense retrieval that misses keyword-exact phrases.
- Score-blended fusion that needs per-query tuning to behave.
- Free-form citations that aren't auditable.
- Generators that confidently make things up when retrieval comes back empty.

This project replaces every one of those with something measurable.

---

## Architecture

```
                   ┌─────────────────────────────────────────────┐
docs/  ──load──►   │  Ingestion Pipeline                         │
mixed              │  • multi-format loader (md / txt / html /pdf)│
formats            │  • chunker (fixed | recursive | semantic)    │
                   │  • cosine dedup ≥ 0.95                       │
                   │  • dual-index (Dense JSON + BM25)            │
                   └────────────────┬────────────────────────────┘
                                    │
                                    ▼
   query  ────►   ┌─────────────────────────────────────────────┐
                  │  Hybrid Retrieval                           │
                  │  • dense top-K (cosine, NumPy)              │
                  │  • sparse top-K (rank_bm25)                 │
                  │  • RRF fusion (k = 60)                      │
                  │  • reranker (LLM | cross-encoder | none)    │
                  └────────────────┬────────────────────────────┘
                                   │ top-N RankedHits
                                   ▼
                  ┌─────────────────────────────────────────────┐
                  │  Generation & Citation                       │
                  │  • grounded prompt (passages + [N] markers)  │
                  │  • IDK hard gate on low retrieval confidence │
                  │  • citation parser + LLM-as-judge verifier   │
                  │  • composite_confidence = w·acc + (1-w)·ret  │
                  └────────────────┬────────────────────────────┘
                                   ▼
                              Answer (text + citations + audit)
```

Five independently testable layers:

1. **Strict Pydantic contracts** — `Document`, `Chunk`, `DenseHit`, `SparseHit`, `FusedHit`, `RankedHit`, `Citation`, `Answer`, `EvalCase`. Every cross-component message is typed.
2. **Ingestion** — markdown / text / HTML (BeautifulSoup) / PDF (pypdf) loaders. Three swappable chunkers (`FixedTokenChunker`, `RecursiveCharacterChunker`, `SemanticChunker`) behind one `Chunker` ABC. Cosine deduplication on insert (≥ 0.95). Dual-index: JSON-backed dense vector store + BM25.
3. **Hybrid retrieval** — dense top-K + sparse top-K → Reciprocal Rank Fusion (configurable `k`) → reranker (default `LLMReranker`, optional `CrossEncoderReranker` via the `[reranker]` extra).
4. **Generation** — grounded prompt that requires bracketed citations on every claim, IDK hard-gate when retrieval confidence is below threshold, citation parser, LLM-as-judge verifier returning per-citation `supported` booleans, composite confidence.
5. **API + CLI + UI** — FastAPI (`POST /v1/ask`, `POST /v1/ingest`, `GET /health`), Click CLI (`rag ingest|ask|eval|serve|config`), optional Streamlit explorer.

---

## Quickstart

```bash
git clone https://github.com/metehanulusoy/rag-hybrid-search
cd rag-hybrid-search
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

export RAG_OPENAI_API_KEY=sk-...

# 1. Ingest a directory (markdown, txt, html, pdf — autodetected).
rag ingest --path ./docs/

# 2. Ask a question.
rag ask --question "What does RRF fusion do?"
# →
# Reciprocal Rank Fusion combines two ranked lists by summing 1/(k+rank), …  [1][2]
#
# [retrieval=0.78 citations=1.00 composite=0.89 idk=False]
#   [1] RRF · ./docs/retrieval/fusion.md
#   [2] Why hybrid search · ./docs/retrieval/why-hybrid.md

# 3. Run the golden eval set.
rag eval --cases eval/golden_qa.jsonl

# 4. Serve.
rag serve   # http://localhost:8100
```

Optional Streamlit UI (`pip install -e ".[ui]"`):

```bash
streamlit run -m rag.ui
```

---

## API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/ask` | Run the full pipeline; returns `Answer`. |
| `POST` | `/v1/ingest` | Ingest a path (file or directory); returns `IngestionReport`. |
| `GET` | `/health` | Liveness + index size. |

`Answer` carries every audit field worth keeping:

```json
{
  "request_id": "...",
  "question": "What does RRF fusion do?",
  "text": "Reciprocal Rank Fusion combines two ranked lists by summing 1/(k+rank), …  [1][2]",
  "citations": [
    {"marker": "[1]", "chunk_id": "chunk_…", "source": "./docs/retrieval/fusion.md", "title": "RRF", "quote": "…"}
  ],
  "used_chunks": [...],
  "retrieval_confidence": 0.78,
  "citation_accuracy": 1.00,
  "composite_confidence": 0.89,
  "is_idk": false,
  "model": "gpt-4o",
  "latency_ms": 1842
}
```

---

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `RAG_OPENAI_API_KEY` | _(required)_ | OpenAI API key. |
| `RAG_EMBEDDING_MODEL` | `text-embedding-3-small` | Embeddings model. |
| `RAG_GENERATION_MODEL` | `gpt-4o` | Grounded answerer model. |
| `RAG_JUDGE_MODEL` | `gpt-4o-mini` | LLM-as-judge for citation verification. |
| `RAG_RERANK_MODEL` | `gpt-4o-mini` | Used when `RAG_RERANK_KIND=llm`. |
| `RAG_CHUNKING_STRATEGY` | `recursive` | `fixed` / `recursive` / `semantic`. |
| `RAG_CHUNK_SIZE_TOKENS` | `512` | Token budget per chunk. |
| `RAG_CHUNK_OVERLAP_TOKENS` | `64` | Overlap; must be `<` chunk size. |
| `RAG_DEDUP_COSINE_THRESHOLD` | `0.95` | Skip ingest if cosine ≥ this against existing chunks. |
| `RAG_DENSE_TOP_K` | `20` | Dense retriever top-K. |
| `RAG_SPARSE_TOP_K` | `20` | Sparse retriever top-K. |
| `RAG_RRF_K` | `60` | Reciprocal Rank Fusion `k`. |
| `RAG_FINAL_TOP_K` | `5` | Chunks passed to the generator after rerank. |
| `RAG_RERANK_KIND` | `llm` | `llm` / `cross-encoder` / `none`. |
| `RAG_IDK_RETRIEVAL_THRESHOLD` | `0.35` | Below this → "I don't know." with no LLM call. |
| `RAG_JUDGE_WEIGHT` | `0.5` | Composite blend: `w·citation_accuracy + (1-w)·retrieval`. |
| `RAG_INDEX_DIR` | `./.rag-index` | Per-strategy index root. |
| `RAG_API_PORT` | `8100` | FastAPI port. |

The strategy is namespaced inside the index dir, so two strategies can coexist on disk and be evaluated head-to-head with the golden Q&A set.

---

## Performance targets

- **Corpus:** 10K–100K chunks. JSON-backed cosine top-K is sub-millisecond at this scale; ChromaDB / Qdrant migration path documented.
- **Query latency P95 < 3 s** (retrieval + rerank + generation). The IDK gate skips the generation roundtrip when retrieval is clearly empty.
- **Faithfulness > 90 %** — measured via the citation verifier; every non-IDK answer reports it as `citation_accuracy`.
- **Citation accuracy > 95 %** on grounded queries — this is the same `citation_accuracy` metric, audited per request.
- **Incremental ingestion** — content-hash skip plus chunk_id stability means re-ingesting an unchanged corpus is a no-op (no embeddings, no LLM calls).
- **Eval suite** — 5-row sample golden set under `eval/`; teams should grow it to 50+ to match the spec target before promotion.

---

## Project standards

- **Versioned chunking strategies** so runs can be compared side-by-side. `Chunker` ABC + per-strategy index directories.
- **Deduplication on ingestion** at cosine ≥ 0.95 — configurable via `RAG_DEDUP_COSINE_THRESHOLD`.
- **Structured citation format** — every claim ends with `[N]`, parsed with a deterministic regex, verified with an LLM-as-judge.
- **Eval suite must pass before deploy** — `rag eval --cases eval/golden_qa.jsonl` returns a non-zero exit if confident-pass falls below an acceptance threshold (callers wire this into CI).
- **"I don't know" is acceptable output** — hard-gated on retrieval confidence; no LLM call when retrieval is clearly empty.
- **Type-safe.** `mypy --strict` with `pydantic.mypy`. `ruff` lint clean.
- **Tested.** `httpx.MockTransport` + deterministic hash-derived pseudo-embeddings; no real network in CI.

---

## Architecture decision records

- [`docs/ADR-001-three-chunking-strategies.md`](docs/ADR-001-three-chunking-strategies.md) — Why three chunkers behind one ABC, swap via config.
- [`docs/ADR-002-rrf-fusion.md`](docs/ADR-002-rrf-fusion.md) — Why Reciprocal Rank Fusion instead of score-normalized linear blending.
- [`docs/ADR-003-confidence-and-idk.md`](docs/ADR-003-confidence-and-idk.md) — Why a hard IDK gate plus a composite confidence score.

---

## License

MIT — see [`LICENSE`](LICENSE).

---

## Acknowledgments

Built collaboratively with **Claude Opus 4.7** as a co-author. Architecture and code review benefited from Anthropic's models throughout.
