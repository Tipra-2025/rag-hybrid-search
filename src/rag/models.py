"""Strict Pydantic v2 contracts for the entire RAG pipeline.

The contracts are intentionally narrow so we never pass dicts between layers:

- `Document` / `Chunk`        — ingestion output
- `DenseHit` / `SparseHit`    — per-store retrieval output
- `FusedHit` / `RankedHit`    — RRF + rerank stages
- `Citation` / `Answer`       — generation output, audit-shaped
- `EvalCase`                  — golden Q&A row used by the eval harness
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ChunkingStrategy(StrEnum):
    """Three chunking strategies the project ships side-by-side for comparison."""

    FIXED = "fixed"
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"


class DocumentFormat(StrEnum):
    MARKDOWN = "markdown"
    TEXT = "text"
    HTML = "html"
    PDF = "pdf"


class Document(BaseModel):
    """Pre-chunking document. `source` is a stable id (path or URL)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source: str = Field(min_length=1)
    format: DocumentFormat
    title: str = ""
    text: str = Field(min_length=1)
    metadata: dict[str, str] = Field(default_factory=dict)


class Chunk(BaseModel):
    """One indexable unit. Stable `chunk_id` allows incremental updates."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk_id: str
    source: str
    title: str = ""
    text: str = Field(min_length=1)
    strategy: ChunkingStrategy
    position: int = Field(ge=0)
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)
    metadata: dict[str, str] = Field(default_factory=dict)


class DenseHit(BaseModel):
    """Result from the dense (embedding) store."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    score: float = Field(ge=-1.0, le=1.0)
    rank: int = Field(ge=1)


class SparseHit(BaseModel):
    """Result from the sparse (BM25) store."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    score: float
    rank: int = Field(ge=1)


class FusedHit(BaseModel):
    """One row of the RRF-fused ranking, before reranking."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    rrf_score: float = Field(ge=0.0)
    dense_rank: int | None = None
    sparse_rank: int | None = None


class RankedHit(BaseModel):
    """Final per-chunk row that the generator sees: chunk + post-rerank score."""

    model_config = ConfigDict(extra="forbid")

    chunk: Chunk
    score: float
    dense_rank: int | None = None
    sparse_rank: int | None = None
    rrf_score: float = 0.0


class Citation(BaseModel):
    """A bracketed citation in the generated answer pointing back to a source chunk."""

    model_config = ConfigDict(extra="forbid")

    marker: str  # e.g. "[1]"
    chunk_id: str
    source: str
    title: str = ""
    quote: str = ""  # short extract from the chunk; empty if not pulled


class Answer(BaseModel):
    """End-to-end response: text + citations + confidence + audit."""

    model_config = ConfigDict(extra="forbid")

    request_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    question: str
    text: str
    citations: list[Citation] = Field(default_factory=list)
    used_chunks: list[RankedHit] = Field(default_factory=list)
    retrieval_confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    citation_accuracy: float = Field(ge=0.0, le=1.0, default=0.0)
    composite_confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    is_idk: bool = False
    model: str = ""
    latency_ms: int = Field(ge=0, default=0)


class EvalCase(BaseModel):
    """Golden Q&A row. The eval harness compares generated `Answer.text` to `expected_answer`."""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    question: str
    expected_answer: str
    must_cite_sources: list[str] = Field(default_factory=list)
    notes: str = ""


class IngestionReport(BaseModel):
    """What the ingestion pipeline did. Counters, not full chunks."""

    model_config = ConfigDict(extra="forbid")

    n_documents: int = 0
    n_chunks_added: int = 0
    n_chunks_deduplicated: int = 0
    n_failures: int = 0
    strategy: ChunkingStrategy
    metadata: dict[str, Any] = Field(default_factory=dict)
