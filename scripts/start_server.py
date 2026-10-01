"""Standalone server startup for PRECISION RAG.

This script starts the FastAPI backend directly, suitable for development.
It reads config from .env or environment variables.

Usage:
    python scripts/start_server.py
    
Or with a dummy API key (for UI testing without LLM calls):
    set RAG_OPENAI_API_KEY=sk-dummy
    python scripts/start_server.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import uvicorn
from rag.api.app import AppState, create_app
from rag.config import load_settings
from rag.engine import RagEngine
from rag.generation import GroundedAnswerer
from rag.ingestion import IngestionPipeline
from rag.llm_client import LLMClient
from rag.models import ChunkingStrategy
from rag.retrieval import HybridRetriever, reranker_for
from rag.services import HealthService, IngestionService, KnowledgeBaseService
from rag.storage import ChunkMetaStore, DocumentStore, KnowledgeBaseStore
from rag.store import BM25Store, DenseVectorStore


def build_state() -> AppState:
    settings = load_settings()
    api_key = settings.openai_api_key or "sk-placeholder-no-llm-calls"

    client = LLMClient(
        api_key=api_key,
        base_url=settings.openai_base_url,
        timeout_s=settings.request_timeout_s,
        max_retries=settings.max_retries,
    )

    dense_path, sparse_path = IngestionPipeline.index_paths(
        settings.index_dir, settings.chunking_strategy
    )
    dense = DenseVectorStore(dense_path)
    sparse = BM25Store(sparse_path)

    ingestion = IngestionPipeline(
        client=client,
        embedding_model=settings.embedding_model,
        dense=dense,
        sparse=sparse,
        strategy=settings.chunking_strategy,
        chunk_size_tokens=settings.chunk_size_tokens,
        overlap_tokens=settings.chunk_overlap_tokens,
        dedup_cosine_threshold=settings.dedup_cosine_threshold,
    )

    reranker = reranker_for(
        settings.rerank_kind if settings.openai_api_key else "none",
        client=client,
        model=settings.rerank_model,
    )
    retriever = HybridRetriever(
        client=client,
        embedding_model=settings.embedding_model,
        dense=dense,
        sparse=sparse,
        reranker=reranker,
        dense_top_k=settings.dense_top_k,
        sparse_top_k=settings.sparse_top_k,
        rrf_k=settings.rrf_k,
        final_top_k=settings.final_top_k,
    )
    answerer = GroundedAnswerer(
        client=client,
        model=settings.generation_model,
        idk_retrieval_threshold=settings.idk_retrieval_threshold,
    )
    engine = RagEngine(
        client=client,
        retriever=retriever,
        answerer=answerer,
        judge_model=settings.judge_model,
        judge_weight=settings.judge_weight,
    )

    data_root = settings.index_dir.parent / "precision-rag-data"
    kb_store = KnowledgeBaseStore(data_root / "knowledge_bases.json")
    doc_store = DocumentStore(data_root)
    chunk_meta = ChunkMetaStore(data_root)
    kb_service = KnowledgeBaseService(
        kb_store=kb_store,
        doc_store=doc_store,
        chunk_meta_store=chunk_meta,
        data_root=data_root,
    )
    ingestion_service = IngestionService(
        kb_service=kb_service,
        doc_store=doc_store,
        chunk_meta_store=chunk_meta,
        llm_client=client,
        embedding_model=settings.embedding_model,
    )
    health_service = HealthService(settings=settings, llm_client=client)

    return AppState(
        client=client,
        engine=engine,
        ingestion=ingestion,
        dense=dense,
        sparse=sparse,
        kb_service=kb_service,
        ingestion_service=ingestion_service,
        health_service=health_service,
    )


if __name__ == "__main__":
    state = build_state()
    app = create_app(state)
    print("\n" + "=" * 60)
    print("  PRECISION RAG — Backend Server")
    print("=" * 60)
    print(f"  API:    http://localhost:8100")
    print(f"  Docs:   http://localhost:8100/docs")
    print(f"  Health: http://localhost:8100/health")
    print("=" * 60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8100, log_level="info")
