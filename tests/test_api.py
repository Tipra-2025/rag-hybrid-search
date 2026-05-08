from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from rag.api.app import AppState, create_app
from rag.engine import RagEngine
from rag.generation import GroundedAnswerer
from rag.ingestion import IngestionPipeline
from rag.llm_client import LLMClient
from rag.models import ChunkingStrategy
from rag.retrieval import HybridRetriever, NoOpReranker
from rag.store import BM25Store, DenseVectorStore

from .conftest import chat_response, embed_response, write_file


@pytest.fixture
def api_client(
    make_llm_client: Callable[[Callable[[httpx.Request], httpx.Response]], LLMClient],
    tmp_path: Path,
) -> TestClient:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/embeddings"):
            body = json.loads(request.content)
            return embed_response(body["input"])
        return chat_response("Stubbed answer with no citations.")

    client = make_llm_client(handler)
    dense = DenseVectorStore(tmp_path / "d.json")
    sparse = BM25Store(tmp_path / "s.json")
    pipeline = IngestionPipeline(
        client=client,
        embedding_model="text-embedding-3-small",
        dense=dense,
        sparse=sparse,
        strategy=ChunkingStrategy.FIXED,
    )
    retriever = HybridRetriever(
        client=client,
        embedding_model="text-embedding-3-small",
        dense=dense,
        sparse=sparse,
        reranker=NoOpReranker(),
        final_top_k=3,
    )
    answerer = GroundedAnswerer(
        client=client, model="gpt-4o", idk_retrieval_threshold=0.0
    )
    engine = RagEngine(
        client=client,
        retriever=retriever,
        answerer=answerer,
        judge_model="gpt-4o-mini",
    )
    state = AppState(
        client=client,
        engine=engine,
        ingestion=pipeline,
        dense=dense,
        sparse=sparse,
    )
    return TestClient(create_app(state))


def test_health(api_client: TestClient) -> None:
    r = api_client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "n_chunks_dense" in body


def test_ingest_then_ask(api_client: TestClient, tmp_path: Path) -> None:
    write_file(tmp_path / "doc.md", "# Alpha\nAlpha is a Greek letter.\n")
    r = api_client.post("/v1/ingest", json={"path": str(tmp_path)})
    assert r.status_code == 200
    body = r.json()
    assert body["n_documents"] == 1
    assert body["n_chunks_added"] >= 1

    r2 = api_client.post("/v1/ask", json={"question": "What is alpha?"})
    assert r2.status_code == 200
    ans = r2.json()
    assert "text" in ans


def test_ask_rejects_empty_question(api_client: TestClient) -> None:
    r = api_client.post("/v1/ask", json={"question": ""})
    assert r.status_code == 422  # Pydantic min_length validation
