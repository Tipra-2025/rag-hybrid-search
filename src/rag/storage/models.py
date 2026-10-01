"""Pydantic models for persistent storage records.

These are separate from the retrieval-layer models in rag.models to keep
the storage layer self-contained and swappable.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class KBStatus(StrEnum):
    ACTIVE = "active"
    INDEXING = "indexing"
    ERROR = "error"


class DocumentStatus(StrEnum):
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"


class RetrievalSettings(BaseModel):
    """Per-knowledge-base retrieval settings. Defaults mirror global Settings."""

    model_config = ConfigDict(extra="forbid")

    chunking_strategy: str = "recursive"
    chunk_size_tokens: int = 512
    chunk_overlap_tokens: int = 64
    dedup_cosine_threshold: float = 0.95
    dense_top_k: int = 20
    sparse_top_k: int = 20
    rrf_k: int = 60
    final_top_k: int = 5
    rerank_kind: str = "llm"
    rerank_model: str = "gpt-4o-mini"
    idk_threshold: float = 0.35
    citation_verification_enabled: bool = True
    judge_weight: float = 0.5


class KnowledgeBaseRecord(BaseModel):
    """Persistent knowledge base metadata."""

    model_config = ConfigDict(extra="forbid")

    id: str
    name: str = Field(min_length=1, max_length=200)
    description: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    document_count: int = 0
    chunk_count: int = 0
    status: KBStatus = KBStatus.ACTIVE
    retrieval_settings: RetrievalSettings = Field(default_factory=RetrievalSettings)
    index_dir: str = ""  # path relative to data_root


class DocumentRecord(BaseModel):
    """Persistent document metadata."""

    model_config = ConfigDict(extra="forbid")

    id: str
    knowledge_base_id: str
    filename: str
    original_path: str = ""
    file_type: str = ""
    file_size: int = 0
    content_hash: str = ""
    page_count: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    status: DocumentStatus = DocumentStatus.UPLOADED
    error_message: str = ""
    chunk_count: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChunkMetaRecord(BaseModel):
    """Metadata stored alongside each chunk in the knowledge base."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    document_id: str
    knowledge_base_id: str
    text: str
    page: int = 0
    position: int = 0
    char_start: int = 0
    char_end: int = 0
    token_count: int = 0
    char_count: int = 0
    source: str = ""
    title: str = ""
    strategy: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
