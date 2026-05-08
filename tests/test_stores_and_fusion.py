from __future__ import annotations

from pathlib import Path

import pytest

from rag.models import Chunk, ChunkingStrategy, DenseHit, SparseHit
from rag.retrieval.fusion import reciprocal_rank_fusion
from rag.store import BM25Store, DenseVectorStore


def _chunk(chunk_id: str, text: str = "lorem ipsum dolor sit amet") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        source="src.md",
        title="src",
        text=text,
        strategy=ChunkingStrategy.FIXED,
        position=0,
        char_start=0,
        char_end=len(text),
    )


def test_dense_store_search_returns_top_k(tmp_path: Path) -> None:
    store = DenseVectorStore(tmp_path / "d.json")
    store.upsert(
        [_chunk("a"), _chunk("b"), _chunk("c")],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.99, 0.05, 0.0]],
    )
    hits = store.search([1.0, 0.0, 0.0], top_k=2)
    assert [h.chunk_id for h in hits] == ["a", "c"]
    assert hits[0].rank == 1


def test_dense_store_persists_and_reloads(tmp_path: Path) -> None:
    path = tmp_path / "d.json"
    s1 = DenseVectorStore(path)
    s1.upsert([_chunk("a")], [[1.0, 0.0, 0.0]])
    s1.save()
    s2 = DenseVectorStore(path)
    assert s2.get("a") is not None
    assert len(s2) == 1


def test_dense_store_dedup_threshold(tmp_path: Path) -> None:
    store = DenseVectorStore(tmp_path / "d.json")
    store.upsert([_chunk("a")], [[1.0, 0.0, 0.0]])
    assert store.is_near_duplicate([1.0, 0.0, 0.0], threshold=0.95) is True
    assert store.is_near_duplicate([0.0, 1.0, 0.0], threshold=0.95) is False


def test_dense_store_upsert_rejects_length_mismatch(tmp_path: Path) -> None:
    store = DenseVectorStore(tmp_path / "d.json")
    with pytest.raises(ValueError):
        store.upsert([_chunk("a")], [])


def test_bm25_store_finds_keyword_match(tmp_path: Path) -> None:
    store = BM25Store(tmp_path / "s.json")
    store.upsert([
        _chunk("a", "Reciprocal rank fusion combines rankings."),
        _chunk("b", "Embedding cosine similarity ranks vectors."),
    ])
    hits = store.search("rank fusion", top_k=5)
    assert hits[0].chunk_id == "a"


def test_bm25_store_empty_query_returns_no_hits(tmp_path: Path) -> None:
    store = BM25Store(tmp_path / "s.json")
    store.upsert([_chunk("a", "non empty text")])
    assert store.search("", top_k=5) == []
    assert store.search("   ", top_k=5) == []


def test_bm25_store_persists(tmp_path: Path) -> None:
    path = tmp_path / "s.json"
    s1 = BM25Store(path)
    s1.upsert([_chunk("a", "alpha bravo charlie")])
    s1.save()
    s2 = BM25Store(path)
    assert len(s2) == 1
    assert s2.search("bravo", top_k=1)


def test_rrf_combines_two_lists() -> None:
    dense = [
        DenseHit(chunk_id="x", score=0.9, rank=1),
        DenseHit(chunk_id="y", score=0.8, rank=2),
    ]
    sparse = [
        SparseHit(chunk_id="y", score=10.0, rank=1),
        SparseHit(chunk_id="z", score=5.0, rank=2),
    ]
    fused = reciprocal_rank_fusion(dense, sparse, k=60)
    ids = [f.chunk_id for f in fused]
    assert "x" in ids and "y" in ids and "z" in ids
    # 'y' appears in both → highest RRF score
    assert fused[0].chunk_id == "y"


def test_rrf_with_disjoint_lists_keeps_both() -> None:
    fused = reciprocal_rank_fusion(
        [DenseHit(chunk_id="x", score=0.9, rank=1)],
        [SparseHit(chunk_id="z", score=10.0, rank=1)],
    )
    assert {f.chunk_id for f in fused} == {"x", "z"}
