"""KnowledgeBaseService: orchestrates KB creation, deletion, and index management.

Each knowledge base gets isolated dense + BM25 indexes under:
    <data_root>/indexes/<kb_id>/<strategy>/dense.json
    <data_root>/indexes/<kb_id>/<strategy>/sparse.json

This ensures documents from one KB never appear in another's results.
"""

from __future__ import annotations

from pathlib import Path

from ..storage import (
    ChunkMetaStore,
    DocumentStore,
    KnowledgeBaseRecord,
    KnowledgeBaseStore,
)
from ..storage.models import KBStatus, RetrievalSettings
from ..store.dense import DenseVectorStore
from ..store.sparse import BM25Store


class KnowledgeBaseService:
    """High-level KB CRUD + per-KB index access."""

    def __init__(
        self,
        kb_store: KnowledgeBaseStore,
        doc_store: DocumentStore,
        chunk_meta_store: ChunkMetaStore,
        data_root: Path,
    ) -> None:
        self._kb_store = kb_store
        self._doc_store = doc_store
        self._chunk_meta = chunk_meta_store
        self._data_root = data_root
        self._index_root = data_root / "indexes"

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def create(self, name: str, description: str = "") -> KnowledgeBaseRecord:
        return self._kb_store.create(name=name, description=description)

    def get(self, kb_id: str) -> KnowledgeBaseRecord | None:
        return self._kb_store.get(kb_id)

    def list_all(self) -> list[KnowledgeBaseRecord]:
        return self._kb_store.list_all()

    def update(self, kb_id: str, **fields: object) -> KnowledgeBaseRecord | None:
        return self._kb_store.update(kb_id, **fields)

    def delete(self, kb_id: str) -> bool:
        """Delete a KB and all its associated indexes and metadata."""
        kb = self._kb_store.get(kb_id)
        if kb is None:
            return False

        # Remove chunk metadata
        self._chunk_meta.delete_all_for_kb(kb_id)

        # Remove document records
        self._doc_store.delete_all_for_kb(kb_id)

        # Remove the index directory
        index_dir = self._index_root / kb_id
        if index_dir.exists():
            import shutil
            shutil.rmtree(index_dir, ignore_errors=True)

        return self._kb_store.delete(kb_id)

    def update_retrieval_settings(
        self, kb_id: str, settings: RetrievalSettings
    ) -> KnowledgeBaseRecord | None:
        return self._kb_store.update(
            kb_id, retrieval_settings=settings.model_dump()
        )

    # ------------------------------------------------------------------
    # Index access
    # ------------------------------------------------------------------

    def index_paths(
        self, kb_id: str, strategy: str = "recursive"
    ) -> tuple[Path, Path]:
        """Return (dense_path, sparse_path) for a given KB + strategy."""
        base = self._index_root / kb_id / strategy
        return (base / "dense.json", base / "sparse.json")

    def get_dense_store(self, kb_id: str, strategy: str = "recursive") -> DenseVectorStore:
        dense_path, _ = self.index_paths(kb_id, strategy)
        return DenseVectorStore(dense_path)

    def get_sparse_store(self, kb_id: str, strategy: str = "recursive") -> BM25Store:
        _, sparse_path = self.index_paths(kb_id, strategy)
        return BM25Store(sparse_path)

    # ------------------------------------------------------------------
    # Aggregated stats
    # ------------------------------------------------------------------

    def get_system_stats(self) -> dict[str, int]:
        """Real system-wide stats across all KBs."""
        kbs = self._kb_store.list_all()
        total_docs = sum(kb.document_count for kb in kbs)
        total_chunks = sum(kb.chunk_count for kb in kbs)
        indexed_docs = 0
        failed_docs = 0
        for kb in kbs:
            docs = self._doc_store.list_for_kb(kb.id)
            for doc in docs:
                if doc.status.value == "indexed":
                    indexed_docs += 1
                elif doc.status.value == "failed":
                    failed_docs += 1
        return {
            "knowledge_bases": len(kbs),
            "total_documents": total_docs,
            "total_chunks": total_chunks,
            "indexed_documents": indexed_docs,
            "failed_documents": failed_docs,
        }
