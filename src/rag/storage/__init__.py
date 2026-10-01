"""Persistent metadata storage for PRECISION RAG.

Provides KnowledgeBaseStore, DocumentStore, and ChunkMetaStore backed by
a JSON flat-file database. Designed so SQLite can be swapped in later without
changing the public API.
"""

from .knowledge_base import KnowledgeBaseStore
from .document_store import DocumentStore
from .chunk_meta import ChunkMetaStore
from .models import (
    KnowledgeBaseRecord,
    DocumentRecord,
    ChunkMetaRecord,
    DocumentStatus,
    KBStatus,
)

__all__ = [
    "KnowledgeBaseStore",
    "DocumentStore",
    "ChunkMetaStore",
    "KnowledgeBaseRecord",
    "DocumentRecord",
    "ChunkMetaRecord",
    "DocumentStatus",
    "KBStatus",
]
