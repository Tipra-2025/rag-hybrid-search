"""Ingestion orchestrator: load → chunk → dedup → dual-index.

Dedup runs *across documents* on the cosine of newly-embedded chunks against
already-indexed chunks. The threshold defaults to 0.95 — above that the chunk
is dropped (treated as a near-exact duplicate). Below, it lands in both stores.

Incremental ingestion is a first-class behavior: re-running the pipeline on
the same corpus is a no-op because every embedded chunk's content hash is
checked against the index before we pay for an embedding.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from ..llm_client import LLMClient
from ..logging import get_logger
from ..models import Chunk, ChunkingStrategy, Document, IngestionReport
from ..store.dense import DenseVectorStore
from ..store.sparse import BM25Store
from .chunkers import SemanticChunker, chunker_for

log = get_logger(__name__)


class IngestionPipeline:
    """Stateful orchestrator. Holds the two stores and the LLM client."""

    def __init__(
        self,
        *,
        client: LLMClient,
        embedding_model: str,
        dense: DenseVectorStore,
        sparse: BM25Store,
        strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE,
        chunk_size_tokens: int = 512,
        overlap_tokens: int = 64,
        semantic_min_tokens: int = 200,
        semantic_max_tokens: int = 900,
        dedup_cosine_threshold: float = 0.95,
        embedding_batch_size: int = 64,
    ):
        self._client = client
        self._embedding_model = embedding_model
        self._dense = dense
        self._sparse = sparse
        self._strategy = strategy
        self._chunk_size_tokens = chunk_size_tokens
        self._overlap_tokens = overlap_tokens
        self._semantic_min = semantic_min_tokens
        self._semantic_max = semantic_max_tokens
        self._dedup_threshold = dedup_cosine_threshold
        self._batch = embedding_batch_size

    async def ingest(self, documents: Iterable[Document]) -> IngestionReport:
        async def embed_batch(texts: list[str]) -> list[list[float]]:
            return await self._client.embed_batch(self._embedding_model, texts)

        chunker = chunker_for(
            self._strategy,
            chunk_size_tokens=self._chunk_size_tokens,
            overlap_tokens=self._overlap_tokens,
            embed_fn=embed_batch,
            semantic_min=self._semantic_min,
            semantic_max=self._semantic_max,
        )

        all_chunks: list[Chunk] = []
        n_docs = 0
        n_failures = 0
        for doc in documents:
            n_docs += 1
            try:
                if isinstance(chunker, SemanticChunker):
                    chunks = await chunker.chunk_async(doc)
                else:
                    chunks = chunker.chunk(doc)
            except Exception as exc:
                log.warning("ingestion.chunk_failed", source=doc.source, error=str(exc))
                n_failures += 1
                continue
            all_chunks.extend(chunks)

        # Drop chunks whose text we have already indexed verbatim.
        existing_ids = self._dense.known_ids()
        fresh = [c for c in all_chunks if c.chunk_id not in existing_ids]

        # Dedup against the dense store via embeddings, batched.
        kept: list[Chunk] = []
        kept_embeddings: list[list[float]] = []
        n_dup = 0
        for i in range(0, len(fresh), self._batch):
            batch = fresh[i : i + self._batch]
            texts = [c.text for c in batch]
            embeds = await self._client.embed_batch(self._embedding_model, texts)
            for chunk, emb in zip(batch, embeds, strict=True):
                if self._dense.is_near_duplicate(emb, threshold=self._dedup_threshold):
                    n_dup += 1
                    continue
                kept.append(chunk)
                kept_embeddings.append(emb)

        if kept:
            self._dense.upsert(kept, kept_embeddings)
            self._sparse.upsert(kept)
            self._dense.save()
            self._sparse.save()

        return IngestionReport(
            n_documents=n_docs,
            n_chunks_added=len(kept),
            n_chunks_deduplicated=n_dup,
            n_failures=n_failures,
            strategy=self._strategy,
            metadata={"embedding_model": self._embedding_model},
        )

    @staticmethod
    def index_paths(base_dir: Path, strategy: ChunkingStrategy) -> tuple[Path, Path]:
        """Return (dense_path, sparse_path) under `base_dir/<strategy>/`."""
        sub = base_dir / strategy.value
        return (sub / "dense.json", sub / "sparse.json")
