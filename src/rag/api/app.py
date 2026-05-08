"""FastAPI factory: /v1/ask, /v1/ingest, /health."""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

from fastapi import FastAPI, Request
from pydantic import BaseModel, Field

from ..engine import RagEngine
from ..ingestion import IngestionPipeline, load_path
from ..llm_client import LLMClient
from ..models import Answer, IngestionReport
from ..store import BM25Store, DenseVectorStore

if TYPE_CHECKING:
    pass


@dataclass
class AppState:
    client: LLMClient
    engine: RagEngine
    ingestion: IngestionPipeline
    dense: DenseVectorStore
    sparse: BM25Store


class AskRequest(BaseModel):
    question: str = Field(min_length=1)


class IngestRequest(BaseModel):
    path: str = Field(min_length=1)


def create_app(state: AppState) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app: FastAPI):  # type: ignore[no-untyped-def]
        try:
            yield
        finally:
            await state.client.aclose()

    app = FastAPI(
        title="rag-hybrid-search",
        version="0.1.0",
        description=(
            "Production-grade hybrid-search RAG. Dense + BM25 + RRF + cross-encoder "
            "rerank + grounded generation with citation verification."
        ),
        lifespan=lifespan,
    )
    app.state.app_state = state

    @app.get("/health", tags=["meta"])
    async def health() -> dict[str, object]:
        return {
            "status": "ok",
            "n_chunks_dense": len(state.dense),
            "n_chunks_sparse": len(state.sparse),
        }

    @app.post("/v1/ask", response_model=Answer, tags=["ask"])
    async def ask(req: AskRequest, request: Request) -> Answer:
        s = cast(AppState, request.app.state.app_state)
        return await s.engine.answer(req.question)

    @app.post("/v1/ingest", response_model=IngestionReport, tags=["ingest"])
    async def ingest(req: IngestRequest, request: Request) -> IngestionReport:
        s = cast(AppState, request.app.state.app_state)
        docs = list(load_path(Path(req.path)))
        return await s.ingestion.ingest(docs)

    return app
