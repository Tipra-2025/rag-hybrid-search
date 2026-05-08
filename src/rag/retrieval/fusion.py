"""Reciprocal Rank Fusion (RRF).

RRF score = Σ 1 / (k + rank_i). The classic Cormack-Clarke choice of k = 60
keeps the contribution of low-ranked hits decaying quickly without zeroing
them out. We expose `k` so teams can tune.

Why RRF over score-normalized linear blending: BM25 scores and cosine scores
live on different scales and have different distributions per query.
Score-blending requires per-query calibration; rank-blending doesn't. RRF is
robust, parameter-light, and matches the public RAG literature consensus.
"""

from __future__ import annotations

from collections.abc import Iterable

from ..models import DenseHit, FusedHit, SparseHit


def reciprocal_rank_fusion(
    dense_hits: Iterable[DenseHit],
    sparse_hits: Iterable[SparseHit],
    *,
    k: int = 60,
) -> list[FusedHit]:
    """Combine two ranked lists into a single RRF-scored list."""
    contrib: dict[str, float] = {}
    dense_rank: dict[str, int] = {}
    sparse_rank: dict[str, int] = {}
    for d_hit in dense_hits:
        dense_rank[d_hit.chunk_id] = d_hit.rank
        contrib[d_hit.chunk_id] = (
            contrib.get(d_hit.chunk_id, 0.0) + 1.0 / (k + d_hit.rank)
        )
    for s_hit in sparse_hits:
        sparse_rank[s_hit.chunk_id] = s_hit.rank
        contrib[s_hit.chunk_id] = (
            contrib.get(s_hit.chunk_id, 0.0) + 1.0 / (k + s_hit.rank)
        )

    fused = [
        FusedHit(
            chunk_id=cid,
            rrf_score=score,
            dense_rank=dense_rank.get(cid),
            sparse_rank=sparse_rank.get(cid),
        )
        for cid, score in contrib.items()
    ]
    fused.sort(key=lambda h: h.rrf_score, reverse=True)
    return fused
