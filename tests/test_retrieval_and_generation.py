from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from rag.engine import RagEngine
from rag.generation import (
    GroundedAnswerer,
    composite_confidence,
    parse_citations,
    verify_citations_async,
)
from rag.ingestion import IngestionPipeline
from rag.llm_client import LLMClient
from rag.models import (
    Chunk,
    ChunkingStrategy,
    Document,
    DocumentFormat,
    RankedHit,
)
from rag.retrieval import HybridRetriever, NoOpReranker, reranker_for
from rag.retrieval.reranker import LLMReranker, _parse_score
from rag.store import BM25Store, DenseVectorStore

from .conftest import chat_json, chat_response, deterministic_embedding, embed_response


def _chunk(cid: str, text: str = "alpha bravo") -> Chunk:
    return Chunk(
        chunk_id=cid,
        source="src.md",
        title="src",
        text=text,
        strategy=ChunkingStrategy.FIXED,
        position=0,
        char_start=0,
        char_end=len(text),
    )


def test_parse_score_clamps_and_handles_garbage() -> None:
    assert _parse_score('{"score": 1.7}') == 1.0
    assert _parse_score('{"score": -0.2}') == 0.0
    assert _parse_score("garbage") == 0.0
    assert _parse_score('{"score": "high"}') == 0.0


def test_composite_confidence_blends() -> None:
    c = composite_confidence(retrieval_confidence=0.4, citation_accuracy=0.8)
    assert c == pytest.approx(0.6, abs=0.01)


def test_composite_confidence_rejects_bad_weight() -> None:
    with pytest.raises(ValueError):
        composite_confidence(
            retrieval_confidence=0.5, citation_accuracy=0.5, judge_weight=1.5
        )


def test_parse_citations_pulls_brackets_and_dedups() -> None:
    hit = RankedHit(chunk=_chunk("c1"), score=0.9)
    citations = parse_citations("Foo [1] is bar [1] but baz [9] is missing.", [hit])
    assert len(citations) == 1
    assert citations[0].chunk_id == "c1"


def test_noop_reranker_keeps_order() -> None:
    import asyncio

    candidates = [
        RankedHit(chunk=_chunk("a"), score=0.5),
        RankedHit(chunk=_chunk("b"), score=0.3),
    ]
    out = asyncio.run(NoOpReranker().rerank(query="q", candidates=candidates, top_k=2))
    assert [c.chunk.chunk_id for c in out] == ["a", "b"]


def test_reranker_for_unknown_kind_raises() -> None:
    with pytest.raises(ValueError):
        reranker_for("totally-not-a-reranker")  # type: ignore[arg-type]


def test_reranker_for_llm_requires_client() -> None:
    with pytest.raises(ValueError):
        reranker_for("llm", client=None)


@pytest.mark.asyncio
async def test_llm_reranker_orders_by_score(
    make_llm_client: Callable[[Callable[[httpx.Request], httpx.Response]], LLMClient],
) -> None:
    seq = iter([0.1, 0.9, 0.5])

    def handler(request: httpx.Request) -> httpx.Response:
        return chat_json({"score": next(seq)})

    client = make_llm_client(handler)
    rer = LLMReranker(client=client, model="gpt-4o-mini")
    candidates = [
        RankedHit(chunk=_chunk("a"), score=0.0),
        RankedHit(chunk=_chunk("b"), score=0.0),
        RankedHit(chunk=_chunk("c"), score=0.0),
    ]
    try:
        out = await rer.rerank(query="q", candidates=candidates, top_k=3)
    finally:
        await client.aclose()
    assert [c.chunk.chunk_id for c in out] == ["b", "c", "a"]


@pytest.mark.asyncio
async def test_hybrid_retriever_returns_post_rerank_top_k(
    make_llm_client: Callable[[Callable[[httpx.Request], httpx.Response]], LLMClient],
    tmp_path: Path,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/embeddings"):
            body = json.loads(request.content)
            return embed_response(body["input"])
        # Reranker call.
        return chat_json({"score": 0.7})

    client = make_llm_client(handler)
    dense = DenseVectorStore(tmp_path / "d.json")
    sparse = BM25Store(tmp_path / "s.json")
    chunks = [
        _chunk("a", "alpha bravo charlie"),
        _chunk("b", "delta echo foxtrot"),
        _chunk("c", "alpha hotel"),
    ]
    dense.upsert(chunks, [deterministic_embedding(c.text) for c in chunks])
    sparse.upsert(chunks)

    retriever = HybridRetriever(
        client=client,
        embedding_model="text-embedding-3-small",
        dense=dense,
        sparse=sparse,
        reranker=NoOpReranker(),
        dense_top_k=5,
        sparse_top_k=5,
        rrf_k=60,
        final_top_k=2,
    )
    try:
        hits = await retriever.retrieve("alpha")
    finally:
        await client.aclose()
    assert len(hits) == 2
    assert all(h.chunk.chunk_id in {"a", "b", "c"} for h in hits)


@pytest.mark.asyncio
async def test_grounded_answerer_short_circuits_on_low_confidence(
    make_llm_client: Callable[[Callable[[httpx.Request], httpx.Response]], LLMClient],
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("LLM should NOT be called when retrieval is below the threshold")

    client = make_llm_client(handler)
    answerer = GroundedAnswerer(
        client=client, model="gpt-4o", idk_retrieval_threshold=0.5
    )
    try:
        ans = await answerer.answer(
            question="q",
            hits=[],
            retrieval_confidence=0.0,
        )
    finally:
        await client.aclose()
    assert ans.is_idk is True
    assert ans.text == "I don't know."


@pytest.mark.asyncio
async def test_grounded_answerer_uses_passages(
    make_llm_client: Callable[[Callable[[httpx.Request], httpx.Response]], LLMClient],
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return chat_response("Alpha is bravo [1].")

    client = make_llm_client(handler)
    answerer = GroundedAnswerer(
        client=client, model="gpt-4o", idk_retrieval_threshold=0.0
    )
    hit = RankedHit(chunk=_chunk("x", "Alpha is bravo, by definition."), score=0.9)
    try:
        ans = await answerer.answer(
            question="What is alpha?",
            hits=[hit],
            retrieval_confidence=0.9,
        )
    finally:
        await client.aclose()
    assert ans.is_idk is False
    assert "[1]" in ans.text


@pytest.mark.asyncio
async def test_verify_citations_returns_accuracy(
    make_llm_client: Callable[[Callable[[httpx.Request], httpx.Response]], LLMClient],
) -> None:
    seq = iter([True, False])

    def handler(request: httpx.Request) -> httpx.Response:
        supported = next(seq)
        return chat_json({"supported": supported, "reason": "..."})

    client = make_llm_client(handler)
    hit = RankedHit(chunk=_chunk("c1", "passage one"), score=0.9)
    hit2 = RankedHit(chunk=_chunk("c2", "passage two"), score=0.8)
    citations = parse_citations("Claim A [1]. Claim B [2].", [hit, hit2])
    try:
        accuracy, _ = await verify_citations_async(
            client=client,
            judge_model="gpt-4o-mini",
            question="Q?",
            answer_text="Claim A [1]. Claim B [2].",
            citations=citations,
            hits=[hit, hit2],
        )
    finally:
        await client.aclose()
    assert accuracy == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_verify_citations_idk_with_no_citations() -> None:
    accuracy, supported = await verify_citations_async(
        client=None,  # type: ignore[arg-type]
        judge_model="x",
        question="q",
        answer_text="I don't know.",
        citations=[],
        hits=[],
    )
    assert accuracy == 1.0
    assert supported == []


@pytest.mark.asyncio
async def test_engine_end_to_end(
    make_llm_client: Callable[[Callable[[httpx.Request], httpx.Response]], LLMClient],
    tmp_path: Path,
) -> None:
    """Wire ingestion → retrieval → generation → verify with stubbed LLM/embeddings."""

    state = {"chat_step": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/embeddings"):
            body = json.loads(request.content)
            return embed_response(body["input"])
        # Sequence of chat calls during answer():
        # 1) reranker per candidate (we use NoOpReranker → 0 calls)
        # 2) grounded answer
        # 3) one verification per citation
        state["chat_step"] += 1
        if state["chat_step"] == 1:
            return chat_response("Alpha rocks [1].")
        return chat_json({"supported": True, "reason": "ok"})

    client = make_llm_client(handler)
    dense = DenseVectorStore(tmp_path / "d.json")
    sparse = BM25Store(tmp_path / "s.json")
    pipeline = IngestionPipeline(
        client=client,
        embedding_model="text-embedding-3-small",
        dense=dense,
        sparse=sparse,
        strategy=ChunkingStrategy.FIXED,
        chunk_size_tokens=2048,
        overlap_tokens=0,
    )

    doc = Document(
        source="alpha.md",
        format=DocumentFormat.MARKDOWN,
        title="Alpha",
        text="Alpha is the first letter and is also a top-tier music release.",
    )
    report = await pipeline.ingest([doc])
    assert report.n_chunks_added >= 1

    answerer = GroundedAnswerer(
        client=client, model="gpt-4o", idk_retrieval_threshold=0.0
    )
    retriever = HybridRetriever(
        client=client,
        embedding_model="text-embedding-3-small",
        dense=dense,
        sparse=sparse,
        reranker=NoOpReranker(),
        dense_top_k=5,
        sparse_top_k=5,
        rrf_k=60,
        final_top_k=3,
    )
    engine = RagEngine(
        client=client,
        retriever=retriever,
        answerer=answerer,
        judge_model="gpt-4o-mini",
        judge_weight=0.5,
    )
    try:
        ans = await engine.answer("What is alpha?")
    finally:
        await client.aclose()
    assert ans.is_idk is False
    assert ans.composite_confidence > 0.0
    assert any(c.marker == "[1]" for c in ans.citations)
