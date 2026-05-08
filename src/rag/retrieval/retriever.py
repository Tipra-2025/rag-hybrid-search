"""End-to-end hybrid retriever: dense + sparse → RRF → rerank → top-K."""

from __future__ import annotations

from ..llm_client import LLMClient
from ..logging import get_logger
from ..models import Chunk, RankedHit
from ..store.dense import DenseVectorStore
from ..store.sparse import BM25Store
from .fusion import reciprocal_rank_fusion
from .reranker import Reranker

log = get_logger(__name__)


class HybridRetriever:
    def __init__(
        self,
        *,
        client: LLMClient,
        embedding_model: str,
        dense: DenseVectorStore,
        sparse: BM25Store,
        reranker: Reranker,
        dense_top_k: int = 20,
        sparse_top_k: int = 20,
        rrf_k: int = 60,
        final_top_k: int = 5,
    ):
        self._client = client
        self._embedding_model = embedding_model
        self._dense = dense
        self._sparse = sparse
        self._reranker = reranker
        self._dense_top_k = dense_top_k
        self._sparse_top_k = sparse_top_k
        self._rrf_k = rrf_k
        self._final_top_k = final_top_k

    async def retrieve(self, query: str) -> list[RankedHit]:
        if not query.strip():
            return []
        embedding = (await self._client.embed_batch(self._embedding_model, [query]))[0]
        dense_hits = self._dense.search(embedding, top_k=self._dense_top_k)
        sparse_hits = self._sparse.search(query, top_k=self._sparse_top_k)
        fused = reciprocal_rank_fusion(dense_hits, sparse_hits, k=self._rrf_k)

        # Resolve fused chunk_ids → Chunk objects, dropping any that disappeared
        # between an index update and this query (rare but possible).
        candidates: list[RankedHit] = []
        for hit in fused[: max(self._final_top_k, self._dense_top_k)]:
            chunk = self._dense.get(hit.chunk_id) or self._sparse.get(hit.chunk_id)
            if chunk is None:
                continue
            candidates.append(
                RankedHit(
                    chunk=chunk,
                    score=hit.rrf_score,
                    dense_rank=hit.dense_rank,
                    sparse_rank=hit.sparse_rank,
                    rrf_score=hit.rrf_score,
                )
            )

        ranked = await self._reranker.rerank(
            query=query, candidates=candidates, top_k=self._final_top_k
        )
        log.info(
            "retrieval.done",
            n_dense=len(dense_hits),
            n_sparse=len(sparse_hits),
            n_fused=len(fused),
            n_final=len(ranked),
        )
        return ranked

    @staticmethod
    def aggregate_confidence(hits: list[RankedHit]) -> float:
        """Mean of post-rerank scores, clamped to [0,1]. Used as retrieval confidence."""
        if not hits:
            return 0.0
        return max(0.0, min(1.0, sum(h.score for h in hits) / len(hits)))

    @staticmethod
    def chunks(hits: list[RankedHit]) -> list[Chunk]:
        return [h.chunk for h in hits]
