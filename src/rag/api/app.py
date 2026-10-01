"""Expanded FastAPI application: KB management + existing /v1/ask and /v1/ingest.

Adds:
  GET  /health                                        - detailed component health
  GET  /knowledge-bases                               - list all KBs
  POST /knowledge-bases                               - create KB
  GET  /knowledge-bases/{kb_id}                       - get KB details
  PATCH /knowledge-bases/{kb_id}                      - update KB name/description
  DELETE /knowledge-bases/{kb_id}                     - delete KB + indexes
  GET  /knowledge-bases/{kb_id}/documents             - list documents
  POST /knowledge-bases/{kb_id}/documents             - upload + ingest document
  GET  /knowledge-bases/{kb_id}/documents/{doc_id}   - get document
  DELETE /knowledge-bases/{kb_id}/documents/{doc_id} - delete document
  POST /knowledge-bases/{kb_id}/documents/{doc_id}/reindex - re-ingest document
  GET  /knowledge-bases/{kb_id}/chunks                - list chunks (with filters)
  POST /knowledge-bases/{kb_id}/ingest                - ingest a file path
  GET  /system/stats                                  - real system stats

Existing endpoints preserved:
  POST /v1/ask    - run the full pipeline (uses default index)
  POST /v1/ingest - ingest a path (uses default index)
"""

from __future__ import annotations

import tempfile
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from fastapi import FastAPI, HTTPException, Query, Request, UploadFile
from fastapi import File as FastAPIFile
from pydantic import BaseModel, Field

from ..engine import RagEngine
from ..ingestion import IngestionPipeline, load_path
from ..llm_client import LLMClient
from ..models import Answer, IngestionReport
from ..services import HealthService, IngestionService, KnowledgeBaseService
from ..storage import (
    ChunkMetaStore,
    DocumentStore,
    KnowledgeBaseStore,
)
from ..storage.models import RetrievalSettings
from ..store import BM25Store, DenseVectorStore

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# App state dataclass
# ---------------------------------------------------------------------------


@dataclass
class AppState:
    client: LLMClient
    engine: RagEngine
    ingestion: IngestionPipeline
    dense: DenseVectorStore
    sparse: BM25Store
    # New KB-aware services (optional for backward compat)
    kb_service: KnowledgeBaseService | None = None
    ingestion_service: IngestionService | None = None
    health_service: HealthService | None = None


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class AskRequest(BaseModel):
    question: str = Field(min_length=1)


class IngestRequest(BaseModel):
    path: str = Field(min_length=1)


class CreateKBRequest(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    description: str = ""


class UpdateKBRequest(BaseModel):
    name: str | None = None
    description: str | None = None


class IngestPathRequest(BaseModel):
    """Ingest a server-side file path into a KB."""
    path: str = Field(min_length=1)
    chunking_strategy: str = "recursive"
    chunk_size_tokens: int = 512
    chunk_overlap_tokens: int = 64


class UpdateSettingsRequest(BaseModel):
    retrieval_settings: RetrievalSettings


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(state: AppState) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app: FastAPI):  # type: ignore[no-untyped-def]
        try:
            yield
        finally:
            await state.client.aclose()

    app = FastAPI(
        title="PRECISION RAG",
        version="1.0.0",
        description=(
            "Reliable Enterprise Retrieval-Augmented Generation. "
            "Dense + BM25 + RRF + cross-encoder rerank + grounded generation "
            "with citation verification. Knowledge base isolation."
        ),
        lifespan=lifespan,
    )
    app.state.app_state = state

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    @app.get("/health", tags=["meta"])
    async def health(request: Request) -> dict[str, Any]:
        s = cast(AppState, request.app.state.app_state)
        n_dense = len(s.dense)
        n_sparse = len(s.sparse)
        if s.health_service:
            return s.health_service.get_health(n_dense, n_sparse)
        return {
            "status": "ok",
            "n_chunks_dense": n_dense,
            "n_chunks_sparse": n_sparse,
        }

    # ------------------------------------------------------------------
    # System stats
    # ------------------------------------------------------------------

    @app.get("/system/stats", tags=["system"])
    async def system_stats(request: Request) -> dict[str, Any]:
        s = cast(AppState, request.app.state.app_state)
        if s.kb_service:
            return s.kb_service.get_system_stats()
        return {
            "knowledge_bases": 0,
            "total_documents": 0,
            "total_chunks": len(s.dense),
            "indexed_documents": 0,
            "failed_documents": 0,
        }

    # ------------------------------------------------------------------
    # Legacy endpoints (backward-compatible)
    # ------------------------------------------------------------------

    @app.post("/v1/ask", response_model=Answer, tags=["ask"])
    async def ask(req: AskRequest, request: Request) -> Answer:
        s = cast(AppState, request.app.state.app_state)
        return await s.engine.answer(req.question)

    @app.post("/v1/ingest", response_model=IngestionReport, tags=["ingest"])
    async def ingest(req: IngestRequest, request: Request) -> IngestionReport:
        s = cast(AppState, request.app.state.app_state)
        docs = list(load_path(Path(req.path)))
        return await s.ingestion.ingest(docs)

    # ------------------------------------------------------------------
    # Knowledge base CRUD
    # ------------------------------------------------------------------

    @app.get("/knowledge-bases", tags=["knowledge-bases"])
    async def list_kbs(request: Request) -> dict[str, Any]:
        s = cast(AppState, request.app.state.app_state)
        if s.kb_service is None:
            raise HTTPException(503, "KB service not initialized")
        kbs = s.kb_service.list_all()
        return {"knowledge_bases": [kb.model_dump(mode="json") for kb in kbs]}

    @app.post("/knowledge-bases", tags=["knowledge-bases"], status_code=201)
    async def create_kb(req: CreateKBRequest, request: Request) -> dict[str, Any]:
        s = cast(AppState, request.app.state.app_state)
        if s.kb_service is None:
            raise HTTPException(503, "KB service not initialized")
        kb = s.kb_service.create(name=req.name, description=req.description)
        return kb.model_dump(mode="json")

    @app.get("/knowledge-bases/{kb_id}", tags=["knowledge-bases"])
    async def get_kb(kb_id: str, request: Request) -> dict[str, Any]:
        s = cast(AppState, request.app.state.app_state)
        if s.kb_service is None:
            raise HTTPException(503, "KB service not initialized")
        kb = s.kb_service.get(kb_id)
        if kb is None:
            raise HTTPException(404, f"Knowledge base {kb_id} not found")
        return kb.model_dump(mode="json")

    @app.patch("/knowledge-bases/{kb_id}", tags=["knowledge-bases"])
    async def update_kb(kb_id: str, req: UpdateKBRequest, request: Request) -> dict[str, Any]:
        s = cast(AppState, request.app.state.app_state)
        if s.kb_service is None:
            raise HTTPException(503, "KB service not initialized")
        updates: dict[str, Any] = {}
        if req.name is not None:
            updates["name"] = req.name
        if req.description is not None:
            updates["description"] = req.description
        kb = s.kb_service.update(kb_id, **updates)
        if kb is None:
            raise HTTPException(404, f"Knowledge base {kb_id} not found")
        return kb.model_dump(mode="json")

    @app.put("/knowledge-bases/{kb_id}/settings", tags=["knowledge-bases"])
    async def update_kb_settings(
        kb_id: str, req: UpdateSettingsRequest, request: Request
    ) -> dict[str, Any]:
        s = cast(AppState, request.app.state.app_state)
        if s.kb_service is None:
            raise HTTPException(503, "KB service not initialized")
        kb = s.kb_service.update_retrieval_settings(kb_id, req.retrieval_settings)
        if kb is None:
            raise HTTPException(404, f"Knowledge base {kb_id} not found")
        return kb.model_dump(mode="json")

    @app.delete("/knowledge-bases/{kb_id}", tags=["knowledge-bases"])
    async def delete_kb(kb_id: str, request: Request) -> dict[str, Any]:
        s = cast(AppState, request.app.state.app_state)
        if s.kb_service is None:
            raise HTTPException(503, "KB service not initialized")
        deleted = s.kb_service.delete(kb_id)
        if not deleted:
            raise HTTPException(404, f"Knowledge base {kb_id} not found")
        return {"deleted": True, "kb_id": kb_id}

    # ------------------------------------------------------------------
    # Document management
    # ------------------------------------------------------------------

    @app.get("/knowledge-bases/{kb_id}/documents", tags=["documents"])
    async def list_documents(kb_id: str, request: Request) -> dict[str, Any]:
        s = cast(AppState, request.app.state.app_state)
        if s.kb_service is None:
            raise HTTPException(503, "KB service not initialized")
        if s.kb_service.get(kb_id) is None:
            raise HTTPException(404, f"Knowledge base {kb_id} not found")
        # Import here to avoid circular imports
        from ..storage import DocumentStore as DS
        doc_store = _get_doc_store(s)
        docs = doc_store.list_for_kb(kb_id)
        return {"documents": [d.model_dump(mode="json") for d in docs]}

    @app.post("/knowledge-bases/{kb_id}/documents", tags=["documents"], status_code=201)
    async def upload_document(
        kb_id: str,
        request: Request,
        file: UploadFile = FastAPIFile(...),
    ) -> dict[str, Any]:
        """Upload a document file and ingest it into the specified KB."""
        s = cast(AppState, request.app.state.app_state)
        if s.kb_service is None or s.ingestion_service is None:
            raise HTTPException(503, "Services not initialized")
        if s.kb_service.get(kb_id) is None:
            raise HTTPException(404, f"Knowledge base {kb_id} not found")
        if not file.filename:
            raise HTTPException(400, "Filename is required")

        # Save upload to temp file
        suffix = Path(file.filename).suffix or ".bin"
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=suffix, prefix="rag_upload_"
        ) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = Path(tmp.name)

        try:
            # Use KB settings for chunking
            kb = s.kb_service.get(kb_id)
            settings = kb.retrieval_settings if kb else None
            result = await s.ingestion_service.ingest_file(
                kb_id=kb_id,
                file_path=tmp_path,
                chunking_strategy=settings.chunking_strategy if settings else "recursive",
                chunk_size_tokens=settings.chunk_size_tokens if settings else 512,
                chunk_overlap_tokens=settings.chunk_overlap_tokens if settings else 64,
                dedup_cosine_threshold=settings.dedup_cosine_threshold if settings else 0.95,
            )
            # Rename the temp file to use original filename for source tracking
            # Update the filename in the result
            if result.get("success") and result.get("doc_id"):
                doc_store = _get_doc_store(s)
                doc_store.update(kb_id, result["doc_id"], filename=file.filename)
            return result
        finally:
            tmp_path.unlink(missing_ok=True)

    @app.get("/knowledge-bases/{kb_id}/documents/{doc_id}", tags=["documents"])
    async def get_document(kb_id: str, doc_id: str, request: Request) -> dict[str, Any]:
        s = cast(AppState, request.app.state.app_state)
        doc_store = _get_doc_store(s)
        doc = doc_store.get(kb_id, doc_id)
        if doc is None:
            raise HTTPException(404, f"Document {doc_id} not found")
        return doc.model_dump(mode="json")

    @app.delete("/knowledge-bases/{kb_id}/documents/{doc_id}", tags=["documents"])
    async def delete_document(kb_id: str, doc_id: str, request: Request) -> dict[str, Any]:
        s = cast(AppState, request.app.state.app_state)
        if s.ingestion_service is None:
            raise HTTPException(503, "Ingestion service not initialized")
        result = await s.ingestion_service.delete_document(kb_id, doc_id)
        if not result.get("success"):
            raise HTTPException(404, result.get("error", "Not found"))
        return result

    @app.post(
        "/knowledge-bases/{kb_id}/documents/{doc_id}/reindex",
        tags=["documents"],
    )
    async def reindex_document(kb_id: str, doc_id: str, request: Request) -> dict[str, Any]:
        s = cast(AppState, request.app.state.app_state)
        if s.ingestion_service is None:
            raise HTTPException(503, "Ingestion service not initialized")
        doc_store = _get_doc_store(s)
        doc = doc_store.get(kb_id, doc_id)
        if doc is None:
            raise HTTPException(404, f"Document {doc_id} not found")
        if not doc.original_path:
            raise HTTPException(400, "Original file path not available for re-index")
        file_path = Path(doc.original_path)
        if not file_path.exists():
            raise HTTPException(400, f"Original file no longer exists: {doc.original_path}")
        # Force re-ingest by deleting and re-ingesting
        await s.ingestion_service.delete_document(kb_id, doc_id)
        kb = s.kb_service.get(kb_id) if s.kb_service else None
        settings = kb.retrieval_settings if kb else None
        return await s.ingestion_service.ingest_file(
            kb_id=kb_id,
            file_path=file_path,
            chunking_strategy=settings.chunking_strategy if settings else "recursive",
            chunk_size_tokens=settings.chunk_size_tokens if settings else 512,
            chunk_overlap_tokens=settings.chunk_overlap_tokens if settings else 64,
        )

    # ------------------------------------------------------------------
    # Chunks
    # ------------------------------------------------------------------

    @app.get("/knowledge-bases/{kb_id}/chunks", tags=["chunks"])
    async def list_chunks(
        kb_id: str,
        request: Request,
        document_id: str | None = Query(default=None),
        page: int | None = Query(default=None),
        search: str | None = Query(default=None),
        limit: int = Query(default=50, ge=1, le=500),
        offset: int = Query(default=0, ge=0),
    ) -> dict[str, Any]:
        s = cast(AppState, request.app.state.app_state)
        if s.kb_service is None:
            raise HTTPException(503, "KB service not initialized")
        if s.kb_service.get(kb_id) is None:
            raise HTTPException(404, f"Knowledge base {kb_id} not found")
        from ..storage import ChunkMetaStore as CMS
        chunk_store = _get_chunk_store(s)
        chunks = chunk_store.list_for_kb(
            kb_id,
            document_id=document_id,
            page=page,
            search=search,
            limit=limit,
            offset=offset,
        )
        total = chunk_store.count_for_kb(kb_id, document_id=document_id)
        return {
            "chunks": [c.model_dump(mode="json") for c in chunks],
            "total": total,
            "limit": limit,
            "offset": offset,
        }

    # ------------------------------------------------------------------
    # KB-scoped ingest (server-side path)
    # ------------------------------------------------------------------

    @app.post("/knowledge-bases/{kb_id}/ingest", tags=["knowledge-bases"])
    async def kb_ingest(
        kb_id: str, req: IngestPathRequest, request: Request
    ) -> dict[str, Any]:
        s = cast(AppState, request.app.state.app_state)
        if s.kb_service is None or s.ingestion_service is None:
            raise HTTPException(503, "Services not initialized")
        if s.kb_service.get(kb_id) is None:
            raise HTTPException(404, f"Knowledge base {kb_id} not found")
        file_path = Path(req.path)
        return await s.ingestion_service.ingest_file(
            kb_id=kb_id,
            file_path=file_path,
            chunking_strategy=req.chunking_strategy,
            chunk_size_tokens=req.chunk_size_tokens,
            chunk_overlap_tokens=req.chunk_overlap_tokens,
        )

    return app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_doc_store(s: AppState) -> "DocumentStore":
    if s.kb_service is None:
        raise HTTPException(503, "KB service not initialized")
    return s.kb_service._doc_store  # type: ignore[attr-defined]


def _get_chunk_store(s: AppState) -> "ChunkMetaStore":
    if s.kb_service is None:
        raise HTTPException(503, "KB service not initialized")
    return s.kb_service._chunk_meta  # type: ignore[attr-defined]
